"""Validate data before downloading teacher, then a real SOURCE-only gradient preflight."""
import argparse
import platform
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch
from common import ROOT, audit_data, catalog, digest, load_json, model_from_config, read_frame, save_json, seed_everything, sha
from flow import FrozenFlowNet2
from losses import five_losses

FLOW_URL = "https://drive.google.com/file/d/1LLYdnObBYzs7rDYYUCGscuC2nwO7y85I/view"


def download_teacher(path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        print("Existing teacher checkpoint found", flush=True); return
    temp = path.with_name(path.name + ".download")
    subprocess.run([sys.executable, "-m", "gdown", "--fuzzy", FLOW_URL, "--continue",
                    "--output", str(temp)], check=True)
    if temp.stat().st_size < 500_000_000:
        raise RuntimeError("FlowNet2 download is too small; refusing HTML/incomplete checkpoint")
    temp.replace(path)


def preflight(config, source, target_root, out, checkpoint, device):
    data = audit_data(source, target_root, config)
    identity = {"config_sha256": digest(config), "release_sha256": sha(ROOT/"release_manifest.json"),
                "data": data, "teacher_sha256": sha(checkpoint)}
    dst = Path(out)/"preflight.json"
    if dst.exists():
        previous = load_json(dst)
        if previous["identity"] != identity or previous["status"] != "PASS":
            raise RuntimeError("Preflight identity changed; use a new output directory")
        print("Matching source preflight reused", flush=True); return
    seed_everything(config["seed"])
    teacher = FrozenFlowNet2(checkpoint, device)
    model = model_from_config(config).to(device).train()
    videos = catalog(source)
    clips = []
    for _, frames in videos[:config["batch_size"]]:
        start = min(64, len(frames)-5)
        clips.append(np.stack([read_frame(p, config["image_size"]) for p in frames[start:start+5]]))
    rgb = torch.from_numpy(np.stack(clips)).to(device)
    start_time = time.monotonic()
    with torch.no_grad():
        motion = torch.cat([teacher(rgb[:, i], rgb[:, i+1]) for i in range(3)], 1)
        true_motion = teacher(rgb[:, 3], rgb[:, 4])
    output = model.forward_train(rgb[:, :4], motion)
    total, parts = five_losses(output, rgb[:, 4], teacher(rgb[:, 3], output["prediction"]), true_motion)
    motion_to_rgb = torch.autograd.grad(parts["motion"], output["prediction"], retain_graph=True)[0]
    if not torch.isfinite(motion_to_rgb).all() or float(motion_to_rgb.norm()) == 0:
        raise RuntimeError("Frozen real FlowNet2 failed to pass L_motion gradients to RGB")
    total.backward()
    grads = {"zero_conv": model.zero.weight.grad, "motion_first_conv": model.motion.blocks[0].layers[0].weight.grad,
             "memory": model.memory.items.grad, "rgb_output": model.decoder.output[0].weight.grad}
    if any(g is None or not torch.isfinite(g).all() or float(g.norm()) == 0 for g in grads.values()):
        raise RuntimeError("Real-source finite/nonzero gradient preflight failed")
    optimizer = torch.optim.Adam(model.parameters(), lr=config["adam_lr"], betas=tuple(config["adam_betas"]))
    old_zero = model.zero.weight.detach().clone(); optimizer.step()
    if torch.equal(old_zero, model.zero.weight) or any(not torch.isfinite(p).all() for p in model.parameters()):
        raise RuntimeError("Source optimizer step failed")
    if any(p.requires_grad or p.grad is not None for p in teacher.parameters()):
        raise RuntimeError("Flow teacher weights must remain frozen")
    model.eval()
    old_state = {k: v.clone() for k, v in model.state_dict().items()}
    with torch.no_grad():
        pred = model.forward_test(rgb[:, :4])
    if pred.shape != rgb[:, 4].shape or not torch.isfinite(pred).all():
        raise RuntimeError("RGB-only test-path preflight failed")
    if any(not torch.equal(v, old_state[k]) for k, v in model.state_dict().items()):
        raise RuntimeError("Inference changed source state")
    save_json(dst, {"status": "PASS", "identity": identity,
        "source_preflight_optimizer_steps": 1, "preflight_model_discarded": True,
        "target_images_accessed_for_training": 0,
        "gradient_norms": {k: float(v.norm()) for k, v in grads.items()},
        "motion_to_rgb_gradient_norm": float(motion_to_rgb.norm()),
        "losses": {k: float(v.detach()) for k, v in parts.items()},
        "elapsed_seconds": time.monotonic()-start_time,
        "environment": {"python": sys.version, "torch": torch.__version__, "numpy": np.__version__,
                         "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(device),
                         "platform": platform.platform()}, "author_confirmed": False})
    print("Real source gradients, teacher and RGB-only inference: PASS", flush=True)


def main():
    p = argparse.ArgumentParser(); p.add_argument("--source", required=True); p.add_argument("--target-root", required=True)
    p.add_argument("--out", required=True); p.add_argument("--checkpoint", required=True); p.add_argument("--device", default="cuda:0")
    args = p.parse_args(); config = load_json(ROOT/"config.json")
    data = audit_data(args.source, args.target_root, config)
    save_json(Path(args.out)/"data_inventory.json", data)
    print("Source and canonical target layout: PASS", flush=True)
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("This production run requires CUDA")
    download_teacher(args.checkpoint)
    preflight(config, args.source, args.target_root, args.out, args.checkpoint, args.device)


if __name__ == "__main__": main()
