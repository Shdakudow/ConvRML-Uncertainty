# Saved-checkpoint prediction exports

| Script | Purpose |
| --- | --- |
| [export_paper_qutcc50_demo.py](export_paper_qutcc50_demo.py) | First 50 fixed RML/normalized DiffuserCam examples using saved epoch-50 QUTCC calibration |
| [export_raw_demo.py](export_raw_demo.py) | Raw/corrupted saved-array packing and GPU refresh/export |

These helpers retain historical `$HOME/ConvRML` data/checkpoint paths. Read
[setup instructions](../../publication_support/docs/RUNNING_EXPERIMENTS.md) before reuse. They do not
train models or deploy the website.

The QUTCC exporter reuses frozen calibration, verifies original sample IDs and
numerical tolerances, and writes arrays/reports under the external project.
The raw helper supports `--study`, `--method`, `--epoch`, `--source-extension`,
`--output-variant` and `--reuse-evaluation`. Without reuse, its GPU refresh path
also recalibrates/evaluates in an isolated directory. Historical defaults remain
clean epoch 50 / corrupted adaptation epoch 10; explicitly select later epochs.

Export consistency checks remain active. A failed check must not be bypassed
to publish different arrays. Authoritative metrics are the completed model's
full-test report. Cluster wrappers, old audits and publishing infrastructure are
preserved in the [archive](../../).
