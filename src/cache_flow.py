"""Precompute SOURCE-only consecutive FlowNet2 pairs, resumable by video."""
import argparse
import hashlib
from pathlib import Path
import shutil
import time
import numpy as np
import torch
from common import ROOT, catalog, digest, load_json, metadata, read_frame, save_json, seed_everything, sha
from flow import FrozenFlowNet2


def main():
    p = argparse.ArgumentParser(); p.add_argument("--source", required=True); p.add_argument("--cache", required=True)
    p.add_argument("--out", required=True); p.add_argument("--checkpoint", required=True); p.add_argument("--device", default="cuda:0")
    args = p.parse_args(); config = load_json(ROOT/"config.json"); seed_everything(config["seed"])
    videos = catalog(args.source); directory = Path(args.cache); directory.mkdir(parents=True, exist_ok=True)
    preflight = load_json(Path(args.out)/"preflight.json")
    identity = {"source_metadata_sha256": digest(metadata(videos)), "teacher_sha256": sha(args.checkpoint),
                "image_size": config["image_size"], "dtype": config["source_flow_cache_dtype"],
                "release_sha256": sha(ROOT/"release_manifest.json")}
    if identity["source_metadata_sha256"] != preflight["identity"]["data"]["source"]["metadata_sha256"]:
        raise RuntimeError("Source changed since preflight")
    identity_path = directory/"identity.json"
    if identity_path.exists() and load_json(identity_path) != identity:
        raise RuntimeError("Incompatible source flow cache")
    save_json(identity_path, identity)
    total_needed = sum((len(fs)-1)*2*config["image_size"]**2*2 for name, fs in videos
                       if not (directory/(name+".json")).exists())
    free = shutil.disk_usage(directory).free
    if free < total_needed*1.05 + 2*1024**3:
        raise RuntimeError(f"Insufficient cache disk: need approximately {total_needed/1024**3:.1f} GiB plus2GiB; free {free/1024**3:.1f} GiB")
    teacher = FrozenFlowNet2(args.checkpoint, args.device)
    records = []
    for name, frames in videos:
        dst, meta = directory/(name+".npy"), directory/(name+".json")
        video_id = digest(metadata([(name, frames)]))
        if meta.exists():
            info = load_json(meta)
            if info["video_metadata_sha256"] != video_id or not dst.is_file() or sha(dst) != info["flow_sha256"]:
                raise RuntimeError("Changed/corrupt completed cache: " + name)
            records.append(info); print("cache reused " + name, flush=True); continue
        shape = (len(frames)-1, 2, config["image_size"], config["image_size"])
        temp = dst.with_name(dst.name+".partial")
        if temp.exists(): temp.unlink()
        cache = np.lib.format.open_memmap(temp, mode="w+", dtype=np.float16, shape=shape)
        started = time.monotonic(); content = hashlib.sha256()
        previous = read_frame(frames[0], config["image_size"])
        content.update(frames[0].name.encode()); content.update(bytes.fromhex(sha(frames[0])))
        with torch.no_grad():
            for start in range(0, len(frames)-1, config["batch_size"]):
                first, second = [], []
                for index in range(start, min(start+config["batch_size"], len(frames)-1)):
                    current = read_frame(frames[index+1], config["image_size"])
                    content.update(frames[index+1].name.encode()); content.update(bytes.fromhex(sha(frames[index+1])))
                    first.append(previous); second.append(current); previous = current
                result = teacher(torch.from_numpy(np.stack(first)).to(args.device),
                                 torch.from_numpy(np.stack(second)).to(args.device)).cpu().numpy()
                if not np.isfinite(result).all() or np.max(np.abs(result)) > np.finfo(np.float16).max:
                    raise FloatingPointError("Invalid/unrepresentable source flow")
                cache[start:start+len(first)] = result.astype(np.float16)
                if start % 500 == 0:
                    print(f"cache {name} {start}/{len(frames)-1}", flush=True)
        cache.flush(); del cache; temp.replace(dst)
        info = {"video": name, "video_metadata_sha256": video_id, "input_content_sha256": content.hexdigest(),
                "flow_sha256": sha(dst), "shape": list(shape), "dtype": "float16", "seconds": time.monotonic()-started}
        save_json(meta, info); records.append(info)
        save_json(Path(args.out)/"state.json", {"status": "CACHING_SOURCE_FLOW", "completed_videos": len(records), "videos": len(videos)})
    record = {"identity": identity, "videos": records, "target_flow_computed": False}
    save_json(directory/"completed.json", record)
    save_json(Path(args.out)/"source_cache_record.json", record)
    print("Completed all source flow caches; no target flow", flush=True)


if __name__ == "__main__": main()
