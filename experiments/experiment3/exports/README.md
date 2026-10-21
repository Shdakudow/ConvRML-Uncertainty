# Frozen-checkpoint inference exports

These are the actual image/bound export utilities captured from Unicorn. They run inference, not model training. The website frontend and its publishing scripts live in the [separate demo repository](https://github.com/Shdakudow/ConvRML-Uncertainty-Demo).

[Complete workflow](../../../docs/RUNNING_EXPERIMENTS.md) · [calibration/clipping definitions](../../../docs/METHODS_AND_METRICS.md) · [source provenance](../../../provenance/README.md)

## Files

| File | Purpose |
| --- | --- |
| `export_paper_qutcc50_demo.py` | First 50 fixed RML/normalized DiffuserCam test images from the epoch-50 current QUTCC checkpoint and its existing calibration. |
| `export_raw_demo.py` | Raw/corrupted exports, with historical default epochs and explicit override options. |
| `audit_epoch20.py` | Numerical audit used to diagnose raw milestone exports. |
| `run.sbatch` / `snapshot.sbatch` | Historical GPU export and CPU saved-array snapshot launchers. Review paths/resources before reuse. |

## Current QUTCC epoch 50

**GPU**, from this directory, with the existing `$HOME/ConvRML` source/data/results layout:

```bash
python export_paper_qutcc50_demo.py --camera rml
python export_paper_qutcc50_demo.py --camera diffuser
```

These reuse `qutcc_paper_style_20261018/results/<camera>/qutcc/evaluation/epoch_050/report.json`. No calibration is fitted on test data. Output goes to `$HOME/ConvRML/results/paper_qutcc50_demo_20261019/<camera>/`:

- `examples.npz`: float32 measurement, target, reconstruction, lower and upper arrays for the first 50 manifest test IDs. Prediction and endpoints are explicitly clipped to `[0,1]`.
- `report.json`: per-image metrics, full-test aggregate reference, fitted q levels, checkpoint hash and learning history.
- `consistency_audit.json`: numerical deviations from original evaluation and explicit tolerances. Original first-ten measurement/target arrays must match exactly; first-ten output maps must agree within `1e-4` normalized intensity. All 50 metric checks remain active.

The export uses the saved checkpoint's inference microbatch. Kernel/GPU differences can produce small float32 deviations. A consistency failure stops export; it does not silently publish a different reconstruction.

## Raw saved arrays and historical export limitations

`export_raw_demo.py --snapshot` packs the **first ten existing evaluation arrays** for the historical clean epoch-50 and corrupted epoch-10 reports. It needs those completed reports/arrays but does not rerun the network. It writes `.bin.gz` images with lossless float32 model arrays and a JSON header.

The `--study`, `--method`, `--epoch`, `--source-extension`, `--output-variant`, and `--reuse-evaluation` options select later raw milestones. Without `--reuse-evaluation`, the GPU refresh path also reruns calibration/full-test evaluation in an isolated refresh directory. Historical defaults remain clean 50 / corrupted 10, not the later clean 70 / corrupted 30.

The historical raw 50-image refresh retains a very strict per-metric tolerance (`2e-5`) and has produced GPU-rounding assertion failures. It is preserved here as exact source, not represented as a repaired final raw exporter. Use the validated full-test `report.json` / saved first-ten `examples.npz` for authoritative raw results, and inspect failed export logs separately from successful model training/evaluation. Never remove consistency checks merely to publish an asset.

Running these exports alone does not commit assets or deploy GitHub Pages.
