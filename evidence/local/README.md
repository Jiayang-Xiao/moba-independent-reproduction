# Local validation, before server execution

These files document implementation checks performed on Windows CPU. They contain
no real dataset performance results and do not certify paper-level reproduction.

- `local_checks.json`: numerical operator checks, full256 student backward,
  actual FlowNet2 network with **random** weights, and synthetic state/resume checks.
- `pipeline_checks.json`: production CLI entrypoints exercised in a private tiny
  copy with an explicitly fake differentiable motion teacher. Cache reuse, training
  interruption and log repair, evaluation alignment and evidence export passed.
- `publication_checks.json`: mocked GitHub API and actual local bare Git push;
  commit verification, upload retry and private-file exclusion passed.

The production package contains no fake estimator. Its real-source preflight must
pass with official FlowNet2 weights before caching or training. CUDA, official weights
and actual datasets were not available for local performance validation.
