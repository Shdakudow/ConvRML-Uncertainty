# Shared checks

From the repository root:

```bash
python -m unittest discover -s tests -v
python tools/verify_source_release.py
```

Full command-line checks require the ML dependencies in `requirements.txt`.
Without them, the ML-only check is skipped.

Experiment-specific model tests are grouped under
[`archive/reproduction/experiments/experiment2/tests/`](../archive/reproduction/experiments/experiment2/tests/). Run from the preserved Exp2 folder:

```bash
cd archive/reproduction/experiments/experiment2
python -m unittest discover -s tests -v
python qutcc/test_protocol.py
```

`test_publication_models.py` checks the six public models, paired loaders and
exact native-interval agreement with the saved-run predictor. The numerical
equivalence report additionally covers initialization, forward/backward and RNG.

Archived Exp3 retains its corruption and continuation tests next to those distinct
implementations. See the [workflow guide](../docs/RUNNING_EXPERIMENTS.md).
