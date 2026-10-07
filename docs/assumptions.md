# Fidelity ledger: paper facts and candidate choices

Technical reference: arXiv2409.17608v1. Final journal full text was unavailable.
All choices are frozen in `config.json`, not selected from this run's target results.

| Component | Verified from accessible paper | Independent choice / unresolved detail |
|---|---|---|
| Task | Normal ShanghaiTech source, Ped2/Avenue cross-domain, no target adaptation | Canonical JPG/flat label layout from existing server; source-test anomalies excluded |
| Inputs | Four RGB inputs,256×256,[-1,1] | PIL bilinear resize, Gaussian reflect border; clean next-frame supervision |
| Blur | SHT kernel3/sigma5 | Same source settings on both targets |
| Encoders/decoder | Figure2 channels64/128/256/512, three MRCA skips, RGB output | Exact conv counts/biases/deconvs match one figure interpretation, not official layer list |
| Zero convolution | Input location illustrated; cited zero-convolution concept | V1 zero-initialized1×1 with identity residual; V0 pure zero-input interpretation had zero motion gradients. Residual is unconfirmed and changes that interpretation |
| MRCA | Reverse conv paths, gates, residual; SHT kernel pairs3/3,3/5,3/7 | High dimension mapped toC256; sharedW1/W2; separate gates;r16 |
| Memory | Ten items, cosine/softmax weighted read; motion train query and appearance test query | Uniform initialization and Adam-only update; no EMA, shrinkage or target update |
| Fusion | Train-only concat1024→512, two1×1convs,Tanh; bypassed at test | Exact module definition retained from V0 figure interpretation |
| Motion modality | FlowNet2 with six encoder channels | Three consecutive2-channel forward flows, raw resized pixel units; source cachefloat16 |
| Motion loss | Future-motion L1 loss stated, RGB-only head shown | Frozen differentiable FlowNet2(lastRGB,predictedRGB), no added motion head; unconfirmed |
| Loss total | Five unweighted loss terms | MSE/L1 mean; gradient horizontal+vertical means; channelwise cosine averaged spatially |
| Optimizer | Adam lr0.0002,cosine annealing | betas.9/.999, ten full epochs,batch4,epochT_max10, no clipping/AMP, final checkpoint |
| Scoring | Patch-max error pyramid→PSNR, min/max, Gaussian temporal filter | Three error-map average levels,16patch/stride1,sigma2,edge border, per-video extrema |
| AUROC | Frame-level AUROC | First4frames discarded, pooled per target; two-target mean derived, AUROC is sole metric |
| Teacher operators | NVIDIA CUDA forward formulas inspected at pinned commit | PyTorch native implementation with numerical tests; warp gradient follows forward floor-based formula rather than old truncation for negative coordinates. No bitwise claim |

This candidate intentionally records all major deviations/assumptions. Making it
trainable does not confirm they were used in the original work. Author feedback
requires a new version identity and fresh baseline, not rewriting old results.
