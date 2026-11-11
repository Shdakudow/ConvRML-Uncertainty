# Automated code checks

This folder contains small CPU tests, **not the held-out image test dataset**.
It is optional for running training, but useful when editing the code.

```bash
python -m unittest discover -s tests -v
```

| File | Checks |
| --- | --- |
| `test_dataset.py` | Image paths, pair order and normalization |
| `test_publication_models.py` | Model APIs, paired-data conventions and weight loading |
| `test_portable_workflow.py` | Training, resume, checkpoint safety, split isolation and evaluation exports |
| `test_paper_calibration.py` | Calibration numerics and separation from test data |
| `test_metrics.py` | Reconstruction and uncertainty metrics |
| `test_release_metadata.py` | Archived checkpoint catalog and published split records |
| `test_learning_curves.py` | Historical learning-curve aggregation |

These fixtures do not retrain benchmark models or guarantee GPU determinism.
Historical verification records describe their recorded revisions.
