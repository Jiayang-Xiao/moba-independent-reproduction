# MoBA independent reproduction candidate V1

**Independent implementation; not official MoBA code or an author-confirmed reproduction.**
This repository implements one fixed paper-guided flow-based candidate, normal-source
training, RGB-only cross-domain testing, and inspectable AUROC evidence. It does not
combine earlier zxVAD experiments or perform target-based parameter searches.

Primary references:

- [MoBA, Knowledge-Based Systems, DOI10.1016/j.knosys.2025.115218](https://doi.org/10.1016/j.knosys.2025.115218).
- [Accessible technical preprint, arXiv2409.17608v1](https://arxiv.org/abs/2409.17608v1).
- [Official NVIDIA FlowNet2 implementation](https://github.com/NVIDIA/flownet2-pytorch/tree/bce69c881ad2b11bef1e6ce9fd0dfd45bd8a1756).

Full journal implementation details and official MoBA code were not available when
this candidate was written. The journal reports SHT→Ped2/Avenue AUROC97.71/87.47;
the accessible preprint flow model reports97.53/87.47. These are literature figures,
not this implementation's results. Results exist only after `evidence/run/summary.json`
is produced by a completed server run.

## Fixed candidate and important assumptions

`config.json` freezes seed17, ten complete source epochs, batch4, four RGB inputs,
256×256, Gaussian3/sigma5, ten memory items, Adam0.0002 and epoch cosine annealing.
All five losses are retained. The exact training budget and scoring constants were
not published in the accessible paper, so these are disclosed engineering choices.

To avoid a dead input path observed in an earlier interpretation, the input adapter
is `motion + zero_initialized_1x1(motion)`. This identity residual is **not confirmed
by the authors**. Six channels mean three consecutive two-channel FlowNet2 flows;
future-motion supervision differentiates a frozen FlowNet2 applied to the last input
and predicted RGB. These interpretations also need author confirmation. See
[all assumptions](docs/assumptions.md), rather than treating them as paper facts.

NVIDIA's network source and checkpoint are retained. A tested PyTorch formula port
replaces its old custom CUDA operators, with attribution and the upstream license.
It is not a replacement flow estimator, and it is not a bitwise replay of old CUDA
forward/backward kernels. The source-only cache stores teacher flows as float16;
training and prediction-side teacher calculations use float32.

## Server workflow

Use the delivered bootstrap command to create a new directory, install an isolated
venv that reuses the existing CUDA PyTorch, authenticate GitHub, and start tmux.
Default server workspace: `/home/xjy/moba-independent-reproduction-v1`.

```bash
bash /home/xjy/moba-independent-reproduction-v1/scripts/launch.sh
```

The detached job checks source/target layout, downloads official FlowNet2 weights,
checks real-source gradients, caches source flows, trains, freezes the final source
checkpoint, tests Ped2/Avenue, recomputes AUROC, and creates/pushes the new public
repository `Jiayang-Xiao/moba-independent-reproduction`. It also publishes inference
weights and a small review bundle as assets of release `moba-independent-v1-seed17`.

Default source:
`/home/xjy/strict_zs_cross_domain_vad/third_party/MPN/data/shanghai/training/frames`.
Default target root:
`/home/xjy/strict_zs_cross_domain_vad/third_party/MPN/data`.
Labels: `frame_labels_ped2.npy`, `frame_labels_avenue.npy`, lexicographic frame order.
Only source normal frames train the model or compute flow. Target labels are used
solely for final AUROC. Target frames never select checkpoints or update memory/BN.

Every target video drops its first four frames. Target sizes are fixed to1962 scored
Ped2 frames and15240 Avenue frames. Scores use three error-pyramid levels,16×16
patch mean with stride1, scale-max sum, negative-log PSNR, per-video min/max inversion,
and Gaussian smoothing sigma2. All constants are frozen before seeing this run's
target AUROC. Each target uses pooled frame AUROC; the two-target average is derived.
No bootstrap intervals are reported and no other metric determines performance.

Source flow caching requires about67GiB for the canonical274,515-frame training set;
reserve around80GiB free for cache, weights and checkpoints. The script checks actual
space. Ten complete epochs are substantially more work than the earlier5000-step
zxVAD runs; no duration or paper-level AUROC is promised.

## Progress, interruption and upload retry

```bash
ROOT=/home/xjy/moba-independent-reproduction-v1
cat "$ROOT/outputs/flow_v1_seed17/state.json"
tail -n 30 "$ROOT/outputs/flow_v1_seed17/execution.log"
tmux has-session -t '=moba-repro-v1' && echo running
```

Detach using Ctrl+b then d. SSH may disconnect after tmux exists and the log is
advancing. Ctrl+c interrupts a foreground command. An ended tmux session alone does
not prove success; require summary/state and publication output.

Rerun `scripts/launch.sh` after a failure. Completed source cache videos are reused,
and `last.pt` restores model, optimizer, scheduler, RNG, epoch and next batch. Code,
config, source and teacher identities must match. Mid-video cache computation restarts
that video; target inference resumes per video. Checkpoint selection remains final epoch.

If training/testing succeeded but publication failed, retry only:

```bash
bash /home/xjy/moba-independent-reproduction-v1/scripts/publish.sh
```

Git2.25-compatible publication uses plain `git init` and an unborn symbolic main ref.
It does not force push or alter another repository's visibility. It stages code,
config, attribution and evidence only; datasets, source flow cache, teacher weights,
virtualenv and full optimizer checkpoints are excluded. An80MB inference checkpoint
is uploaded via GitHub Release. Actual paths remain in server evidence as provenance.

## Validation scope and loading inference weights

Local tests include NumPy operator oracles, numerical gradients, frozen-state RGB
inference, a serialized interruption/resume fixture, full256 student backward and the
actual162,518,834-parameter FlowNet2 architecture with random weights. Local CPU tests
do not verify pretrained weight quality, real datasets, GPU speed or AUROC. The server
preflight additionally verifies the real downloaded teacher and real normal-source clips.

Private synthetic CLI checks also cover source-cache reuse, mid-epoch interruption,
repair of an interrupted final log line, complete fixed-epoch training, frame/label
alignment, cached target evaluation and independent result export. Offline publication
checks use a local bare Git remote and a mocked GitHub API. They verify commit identity,
upload retry and exclusion of private files; they do not constitute a real GitHub push.
Local check records are in `evidence/local/`, with their limited scope stated explicitly.

```python
import sys, torch
sys.path.insert(0, "src")
from common import model_from_config
saved = torch.load("model.pt", map_location="cpu", weights_only=True)
model = model_from_config(saved["config"]).eval()
model.load_state_dict(saved["model"])
# model.forward_test(rgb) accepts B,4,3,256,256 RGB in [-1,1]; no FlowNet2 needed at test.
```

Until original implementation details are confirmed, results must be described as
this disclosed independent candidate, not an official MoBA replication or SOTA gain.
