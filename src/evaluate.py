"""Freeze final source fit before RGB-only Ped2/Avenue inference; report AUROC only."""
import argparse
from collections import deque
from pathlib import Path
import numpy as np
import torch
from common import ROOT, audit_data, catalog, digest, load_json, model_from_config, read_frame, save_json, seed_everything, sha
from scoring import auroc, pyramid_psnr, video_anomaly_scores


def main():
    p = argparse.ArgumentParser(); p.add_argument("--source", required=True); p.add_argument("--target-root", required=True)
    p.add_argument("--out", required=True); p.add_argument("--device", default="cuda:0")
    args = p.parse_args(); config = load_json(ROOT/"config.json"); seed_everything(config["seed"])
    out = Path(args.out); trained = load_json(out/"trained.json")
    data = audit_data(args.source, args.target_root, config)
    if (trained["status"] != "TRAINED" or trained["epoch_completed"] != config["epochs"]
            or trained["context"]["config_sha256"] != digest(config)
            or trained["context"]["release_sha256"] != sha(ROOT/"release_manifest.json")
            or trained["context"]["source_metadata_sha256"] != data["source"]["metadata_sha256"]
            or trained["checkpoint_sha256"] != sha(out/"last.pt")):
        raise RuntimeError("Changed/incomplete final source fit")
    preflight = load_json(out/"preflight.json")
    if data != preflight["identity"]["data"]: raise RuntimeError("Dataset changed since source preflight")
    freeze = {"config": config, "trained": trained, "data": data,
              "preflight_sha256": sha(out/"preflight.json"), "target_input": "RGB only"}
    if (out/"evaluation_freeze.json").exists() and load_json(out/"evaluation_freeze.json") != freeze:
        raise RuntimeError("Evaluation freeze changed")
    save_json(out/"evaluation_freeze.json", freeze)
    model = model_from_config(config).to(args.device).eval()
    saved = torch.load(out/"last.pt", map_location=args.device, weights_only=False)
    if saved["context"] != trained["context"]: raise RuntimeError("Checkpoint context mismatch")
    model.load_state_dict(saved["model"]); del saved
    memory_before = model.memory.items.detach().clone()
    bn_before = {name: value.clone() for name, value in model.named_buffers()}
    calls = {"motion": 0, "fusion": 0}
    def hook_motion(module, ins, outs): calls["motion"] += 1
    def hook_fusion(module, ins, outs): calls["fusion"] += 1
    hooks = [model.motion.register_forward_hook(hook_motion), model.fusion.register_forward_hook(hook_fusion)]
    rows = []
    for target, expected in config["targets"].items():
        target_files = catalog(Path(args.target_root)/target/"testing/frames")
        y = np.load(Path(args.target_root)/("frame_labels_"+target+".npy"), allow_pickle=False).reshape(-1)
        video_dir = out/"video_scores"/target; video_dir.mkdir(parents=True, exist_ok=True)
        all_psnr, all_score, all_labels, all_video, all_frame = [], [], [], [], []
        offset = 0
        with torch.inference_mode():
            for vid, (name, files) in enumerate(target_files):
                npz, meta = video_dir/(name+".npz"), video_dir/(name+".json")
                if npz.exists() and meta.exists():
                    record = load_json(meta)
                    if record["freeze_sha256"] != sha(out/"evaluation_freeze.json") or record["sha256"] != sha(npz):
                        raise RuntimeError("Changed per-video scores")
                    with np.load(npz, allow_pickle=False) as value:
                        psnr, score = value["psnr"].copy(), value["score"].copy()
                else:
                    window = deque([read_frame(p, config["image_size"]) for p in files[:4]], maxlen=4)
                    raw = []
                    for frame in range(4, len(files)):
                        current = read_frame(files[frame], config["image_size"])
                        inputs = torch.from_numpy(np.stack(window)[None]).to(args.device)
                        prediction = model.forward_test(inputs)
                        target_rgb = torch.from_numpy(current[None]).to(args.device)
                        value = pyramid_psnr(prediction, target_rgb, config["score"]["pyramid_levels"],
                                             config["score"]["patch"], config["score"]["stride"])
                        raw.append(float(value.item())); window.append(current)
                    psnr = np.asarray(raw, dtype=np.float64)
                    score = video_anomaly_scores(psnr, config["score"]["smoothing_sigma"])
                    with npz.open("wb") as f: np.savez_compressed(f, psnr=psnr, score=score)
                    save_json(meta, {"sha256": sha(npz), "freeze_sha256": sha(out/"evaluation_freeze.json")})
                if len(psnr) != len(files)-4 or len(score) != len(psnr): raise RuntimeError("Frame score alignment mismatch")
                all_psnr.extend(psnr); all_score.extend(score); all_labels.extend(y[offset+4:offset+len(files)])
                all_video.extend([vid]*len(score)); all_frame.extend(range(4, len(files)))
                offset += len(files)
                print(f"evaluate {target} {name} {len(all_score)} frames", flush=True)
        if len(all_score) != expected["scored_frames"]: raise RuntimeError("Scored-frame total mismatch")
        dst = out/(target+".npz")
        with dst.open("wb") as f:
            np.savez_compressed(f, psnr=np.asarray(all_psnr), score=np.asarray(all_score), label=np.asarray(all_labels),
                video=np.asarray(all_video), frame=np.asarray(all_frame), video_names=np.asarray([name for name, _ in target_files]))
        auc = auroc(all_labels, all_score)
        record = {"target": target, "frames": len(all_score), "AUROC": auc,
                  "scores_sha256": sha(dst), "freeze_sha256": sha(out/"evaluation_freeze.json")}
        save_json(out/(target+".json"), record); rows.append(record)
    if calls != {"motion": 0, "fusion": 0} or not torch.equal(memory_before, model.memory.items):
        raise RuntimeError("Target inference executed motion/fusion or changed memory")
    if any(not torch.equal(value, bn_before[name]) for name, value in model.named_buffers()):
        raise RuntimeError("Target inference changed source BatchNorm")
    for hook in hooks: hook.remove()
    with (out/"results.csv").open("w", encoding="utf-8") as f:
        f.write("study,seed,target,frames,AUROC\n")
        for row in rows: f.write(f"{config['identity']},{config['seed']},{row['target']},{row['frames']},{row['AUROC']:.17g}\n")
    summary = {"status": "COMPLETED", "identity": config["identity"], "author_confirmed": False,
               "metric": "AUROC only", "targets": rows, "macro_AUROC": float(np.mean([r["AUROC"] for r in rows])),
               "target_motion_calls": calls, "checkpoint_sha256": trained["checkpoint_sha256"],
               "evaluation_freeze_sha256": sha(out/"evaluation_freeze.json"),
               "interpretation": "Provisional independent MoBA-style baseline; no official-reproduction or SOTA claim"}
    save_json(out/"summary.json", summary); save_json(out/"state.json", summary)
    print(json_summary(summary), flush=True)


def json_summary(value):
    import json
    return json.dumps(value, indent=2)


if __name__ == "__main__": main()
