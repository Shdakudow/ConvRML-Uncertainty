# Changes

## 0.1.0

- Four annotated Exp2 models and two Exp3 models, with common layers implemented once.
- Portable training, epoch-boundary resume, calibration and held-out evaluation.
- Original model parameter names retained; existing weights remain compatible.
- Explicit empirical, reference and native calibration modes, with separate raw/clipped metrics.
- Ordered split IDs, settings, seed policies, verified checkpoint downloads and environment records.
- Cluster/job/demo tooling separated under archive; no cluster dependency in the public workflow.
- Software citation separated from upstream paper attribution.
