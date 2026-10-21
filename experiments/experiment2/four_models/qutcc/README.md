# Current QUTCC within the four-model comparison

Use the parent [four-model README](../README.md) and `../run_model.py --method qutcc`. This is the current original-style implementation: ConvNeXt/FPN, 48 FiLM sites, uniform per-image q, pinball-only training and independent quantile-space calibration.

Read `model.py` for architecture/loss, `train.py` for optimizer/checkpointing, `data.py` for paired split loading, `calibration.py` for Algorithm 1, and `evaluate.py` for frozen calibration/test inference. `protocol.json` records settings; `provenance.json` verifies exact copies from the previously published current QUTCC snapshot. No trained weights are bundled here.

The common entry point maps RML to task 0 and normalized DiffuserCam to task 1. Direct source retains raw task 2 (QUTCC) / task 3 (Im2im) for compatibility, but the common four-model normal-camera comparison does not launch raw/adaptation jobs.

Dataset lookup uses `CONVRML_PROJECT`; outputs are written under this directory's `results/`. Calibration/evaluation may load historical compact state dictionaries when data identities match. Exact historical training resume requires the original source/full-state checkpoint; formatting or relocating a training source tree changes its source hash.
