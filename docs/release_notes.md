Independent paper-guided MoBA flow candidate V1, seed17. This release is not official
code and has not been confirmed by the authors. Source-only ShanghaiTech training,
final-epoch checkpoint, RGB-only Ped2/Avenue evaluation. AUROC is the sole metric.

The repository records architecture, future-motion supervision, source-flow cache,
training-budget and scoring assumptions. See docs/assumptions.md before comparing
results with the journal or preprint.

Assets: model.pt contains inference state/config (torch.load with weights_only=True);
review_bundle.zip contains hashes, raw scores, AUROC, source-only preflight, training
records and environment. SHA256 is recorded in evidence/run/model_release.json.
No dataset, FlowNet2 teacher weights or source flow cache is redistributed.
