# Verification

```bash
python -m pip install -e '.[dev]'
python -m unittest discover -s tests -v
```

The public tests use small temporary paired fixtures. They check model/data APIs,
calibration numerics, exact uninterrupted-versus-resumed optimizer/RNG progress,
strict checkpoint identity, split leakage, immutable results, saved examples,
checkpoint indexes and release metadata. They do not allocate a GPU or train the
published models on the full benchmark.
The paired-file DLMD check also trains both method fixtures, strictly reloads
their checkpoints, calibrates on separate IDs and exports test metrics/examples.
It uses the real Im2im/QUTCC loss methods with tiny surrogate networks, not the
full ConvNeXt backbone or benchmark data.

Historical source/cluster checks are optional and separate:

```bash
python -m unittest discover -s archive/operations/tests -v
python -m unittest discover -s archive/reproduction/tests -v
python -m unittest discover -s archive/publication_support/calibration_legacy/tests -v
python archive/publication_support/tools/verify_source_release.py
```

[Verification record](../reproducibility/verification.json) states the tested
environment and scope. CUDA numerical equality across hardware is not established
by CPU fixture checks.
