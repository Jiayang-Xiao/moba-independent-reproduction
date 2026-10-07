"""Ten full SOURCE epochs, final checkpoint only; resumable mid-epoch."""
import argparse
from collections import OrderedDict
import json
import math
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from common import ROOT, catalog, digest, load_json, metadata, model_from_config, read_frame, save_json, seed_everything, sha
from flow import FrozenFlowNet2
from losses import five_losses


class SourceWindows(Dataset):
    def __init__(self, videos, cache_dir, size):
        self.videos, self.cache_dir, self.size = videos, Path(cache_dir), size
        self.ends = np.cumsum([len(fs)-4 for _, fs in videos])
        self.open_cache = OrderedDict()

    def __len__(self):
        return int(self.ends[-1])

    def __getitem__(self, index):
        v = int(np.searchsorted(self.ends, index, side="right"))
        offset = int(index - (self.ends[v-1] if v else 0))
        name, files = self.videos[v]
        if name not in self.open_cache:
            if len(self.open_cache) >= 2:
                self.open_cache.popitem(last=False)
            self.open_cache[name] = np.load(self.cache_dir/(name+".npy"), mmap_mode="r", allow_pickle=False)
        self.open_cache.move_to_end(name)
        flow = self.open_cache[name]
        rgb = np.stack([read_frame(p, self.size) for p in files[offset:offset+5]])
        # Three two-channel input pairs; fourth pair supervises future motion.
        motion = np.array(flow[offset:offset+3], dtype=np.float32).reshape(6, self.size, self.size)
        future_flow = np.array(flow[offset+3], dtype=np.float32)
        return torch.from_numpy(rgb), torch.from_numpy(motion), torch.from_numpy(future_flow)


def update(model, teacher, optimizer, rgb, motion, future_flow):
    optimizer.zero_grad(set_to_none=True)
    output = model.forward_train(rgb[:, :4], motion)
    predicted_flow = teacher(rgb[:, 3], output["prediction"])
    total, parts = five_losses(output, rgb[:, 4], predicted_flow, future_flow)
    total.backward()
    for parameter in model.parameters():
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            raise FloatingPointError("Nonfinite source gradient; checkpoint retained")
    norm = float(model.zero.weight.grad.norm()) if model.zero.weight.grad is not None else 0
    first_norm = float(model.motion.blocks[0].layers[0].weight.grad.norm()) \
        if model.motion.blocks[0].layers[0].weight.grad is not None else 0
    # Real-data preflight established nonzero connectivity. Static clips may have zero signal later.
    if not math.isfinite(norm) or not math.isfinite(first_norm):
        raise FloatingPointError("Invalid motion-path gradient")
    optimizer.step()
    return {k: float(v.detach()) for k, v in parts.items()} | {
        "total": float(total.detach()), "zero_conv_grad": norm, "motion_first_grad": first_norm}


def snapshot(model, optimizer, scheduler, context, epoch, next_batch, step):
    return {"context": context, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "epoch": epoch, "next_batch": next_batch, "step": step,
            "rng": {"python": random.getstate(), "numpy": np.random.get_state(),
                    "torch": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}}


def write_checkpoint(path, checkpoint):
    path = Path(path); temp = path.with_name(path.name+".tmp")
    with temp.open("wb") as f:
        torch.save(checkpoint, f); f.flush(); os.fsync(f.fileno())
    temp.replace(path)


def restore_rng(rng):
    random.setstate(rng["python"]); np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch"].cpu())
    if torch.cuda.is_available(): torch.cuda.set_rng_state_all([s.cpu() for s in rng["cuda"]])


def main():
    p = argparse.ArgumentParser(); p.add_argument("--source", required=True); p.add_argument("--cache", required=True)
    p.add_argument("--out", required=True); p.add_argument("--checkpoint", required=True); p.add_argument("--device", default="cuda:0")
    args = p.parse_args(); config = load_json(ROOT/"config.json"); seed_everything(config["seed"])
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    preflight = load_json(out/"preflight.json"); cache_record = load_json(Path(args.cache)/"completed.json")
    videos = catalog(args.source); source_id = digest(metadata(videos))
    if (preflight["status"] != "PASS" or source_id != cache_record["identity"]["source_metadata_sha256"]
            or sha(args.checkpoint) != cache_record["identity"]["teacher_sha256"]):
        raise RuntimeError("Changed/missing source preflight or cache identity")
    context = {"identity": config["identity"], "config_sha256": digest(config),
               "release_sha256": sha(ROOT/"release_manifest.json"), "source_metadata_sha256": source_id,
               "cache_record_sha256": sha(Path(args.cache)/"completed.json"),
               "preflight_sha256": sha(out/"preflight.json"), "teacher_sha256": sha(args.checkpoint)}
    save_json(out/"config.json", config)
    last, done = out/"last.pt", out/"trained.json"
    if done.exists():
        completed = load_json(done)
        if completed["context"] != context or completed["checkpoint_sha256"] != sha(last):
            raise RuntimeError("Changed completed training identity")
        print("Completed matching source fit reused", flush=True); return
    teacher = FrozenFlowNet2(args.checkpoint, args.device)
    model = model_from_config(config).to(args.device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["adam_lr"], betas=tuple(config["adam_betas"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"], eta_min=0)
    epoch, next_batch, step = 0, 0, 0
    if last.exists():
        saved = torch.load(last, map_location=args.device, weights_only=False)
        if saved["context"] != context: raise RuntimeError("Checkpoint protocol/source identity mismatch")
        model.load_state_dict(saved["model"]); optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        epoch, next_batch, step = saved["epoch"], saved["next_batch"], saved["step"]
        restore_rng(saved["rng"]); del saved
        print(f"Resume epoch{epoch+1}, next batch{next_batch}, step{step}", flush=True)
    dataset = SourceWindows(videos, args.cache, config["image_size"])
    per_epoch = math.ceil(len(dataset)/config["batch_size"])
    total_steps = per_epoch * config["epochs"]
    logs = out/"train.jsonl"
    # Discard records beyond the atomic checkpoint after a crash; no duplicate step identities.
    if logs.exists():
        lines = logs.read_text(encoding="utf-8").splitlines()
        kept = []
        for index, line in enumerate(lines):
            try:
                logged_step = json.loads(line)["step"]
            except (json.JSONDecodeError, KeyError):
                if index != len(lines)-1:
                    raise RuntimeError("Corrupt training log before its final line")
                # A crash can interrupt the final line write, while last.pt remains valid.
                break
            if logged_step <= step:
                kept.append(line)
        logs.write_text("".join(line+"\n" for line in kept), encoding="utf-8")
    started = time.monotonic()
    with logs.open("a", encoding="utf-8", buffering=1) as log:
        while epoch < config["epochs"]:
            order = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(config["seed"]+epoch)).tolist()
            batches = [order[i:i+config["batch_size"]] for i in range(0, len(order), config["batch_size"])]
            loader = DataLoader(dataset, batch_sampler=batches[next_batch:], num_workers=config["workers"],
                pin_memory=args.device.startswith("cuda"), generator=torch.Generator().manual_seed(50000+epoch),
                persistent_workers=False)
            for batch, (rgb, motion, future_flow) in enumerate(loader, start=next_batch):
                stats = update(model, teacher, optimizer, rgb.to(args.device, non_blocking=True),
                               motion.to(args.device, non_blocking=True), future_flow.to(args.device, non_blocking=True))
                step += 1; next_batch = batch+1
                if step == 1 or step % config["log_every_steps"] == 0:
                    row = {"step": step, "epoch": epoch+1, "batch": next_batch, "lr": optimizer.param_groups[0]["lr"],
                           "elapsed_this_process": time.monotonic()-started, **stats}
                    log.write(json.dumps(row)+"\n"); print("train " + json.dumps(row), flush=True)
                if step % config["checkpoint_every_steps"] == 0:
                    write_checkpoint(last, snapshot(model, optimizer, scheduler, context, epoch, next_batch, step))
                    save_json(out/"state.json", {"status": "TRAINING", "step": step, "total_steps": total_steps,
                                               "epoch": epoch+1, "next_batch": next_batch})
            scheduler.step(); epoch += 1; next_batch = 0
            write_checkpoint(last, snapshot(model, optimizer, scheduler, context, epoch, next_batch, step))
            save_json(out/"state.json", {"status": "TRAINING", "step": step, "total_steps": total_steps, "epoch_completed": epoch})
        if step != total_steps: raise RuntimeError("Did not consume exactly all source windows in every epoch")
    completed = {"status": "TRAINED", "context": context, "epoch_completed": epoch, "step": step,
                 "windows_per_epoch": len(dataset), "batch_size": config["batch_size"],
                 "checkpoint_sha256": sha(last), "selection": "final source epoch; no target selection"}
    save_json(done, completed); save_json(out/"state.json", completed)
    print("Completed fixed source training: " + str(done), flush=True)


if __name__ == "__main__": main()
