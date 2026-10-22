# Shared checks

From the repository root:

```bash
python -m unittest discover -s tests -v
python tools/verify_source_release.py
```

Full command-line checks require the ML dependencies in `requirements.txt`.
Without them, the ML-only check is skipped.

Experiment-specific model tests are grouped under
[`experiments/experiment2/tests/`](../experiments/experiment2/tests/). Run from Exp2:

```bash
cd experiments/experiment2
python -m unittest discover -s tests -v
python qutcc/test_protocol.py
```

Exp3 retains its corruption and continuation tests next to those distinct
implementations. See the [workflow guide](../docs/RUNNING_EXPERIMENTS.md).
