"""Atomic records and declared source/target identities."""
import hashlib
import json
import os
from pathlib import Path
import random
import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, ensure_ascii=False); f.write("\n"); f.flush(); os.fsync(f.fileno())
    tmp.replace(path)


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # Native gather/scatter and the original atomic CUDA warp are not bitwise deterministic.
    torch.use_deterministic_algorithms(False)


def catalog(path):
    root = Path(path).resolve()
    if not root.is_dir():
        raise FileNotFoundError("Frames directory missing: " + str(root))
    videos = [(v.name, sorted(v.glob("*.jpg"))) for v in sorted(root.iterdir()) if v.is_dir()]
    if not videos or any(len(fs) < 5 for _, fs in videos):
        raise RuntimeError("Every video must have at least five JPG frames: " + str(root))
    return videos


def metadata(videos):
    return [{"video": name, "frames": [{"name": p.name, "bytes": p.stat().st_size,
             "mtime_ns": p.stat().st_mtime_ns} for p in fs]} for name, fs in videos]


def read_frame(path, size=256):
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR),
                           dtype=np.float32) / 127.5 - 1
    return np.ascontiguousarray(array.transpose(2, 0, 1))


def model_from_config(config):
    from model import MoBACandidate
    return MoBACandidate(channels=tuple(config["channels"]), memory_items=config["memory_items"])


def audit_data(source, target_root, config):
    source, target_root = Path(source).resolve(), Path(target_root).resolve()
    if "training" not in source.parts:
        raise RuntimeError("Source path must explicitly select a normal training split")
    videos = catalog(source)
    if len(videos) != config["source_expected_videos"]:
        raise RuntimeError("Expected330 ShanghaiTech normal training videos")
    source_meta = metadata(videos)
    record = {"source": {"root": str(source), "videos": len(videos),
                          "frames": sum(len(fs) for _, fs in videos),
                          "windows": sum(len(fs)-4 for _, fs in videos),
                          "metadata_sha256": digest(source_meta)}, "targets": {}}
    for name, expected in config["targets"].items():
        target = (target_root / name / "testing/frames").resolve()
        if source == target or source in target.parents or target in source.parents:
            raise RuntimeError("Source and target paths overlap")
        cat = catalog(target)
        label_path = target_root / ("frame_labels_" + name + ".npy")
        labels = np.load(label_path, allow_pickle=False).reshape(-1)
        nframes = sum(len(fs) for _, fs in cat)
        nscored = sum(len(fs)-4 for _, fs in cat)
        if len(cat) != expected["videos"] or nscored != expected["scored_frames"] or len(labels) != nframes:
            raise RuntimeError("Canonical target/frame/label mismatch: " + name)
        if not np.isin(labels, [0, 1]).all():
            raise RuntimeError("Nonbinary target labels: " + name)
        record["targets"][name] = {"root": str(target), "videos": len(cat), "frames": nframes,
            "scored_frames": nscored, "metadata_sha256": digest(metadata(cat)), "label_sha256": sha(label_path)}
    return record
