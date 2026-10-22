# Shared implementations

Identical Exp2/Exp3 code is stored once here. Files at the original experiment
locations are **relative symbolic links**, not copied implementations. Review the
files below; launch commands from their experiment entry points as before.

| Shared file | Used for |
| --- | --- |
| [models/backbone.py](models/backbone.py) | ConvNeXt/FPN backbone for all current models |
| [models/film.py](models/film.py) | Quantile embedding projections and 48 FiLM sites |
| [models/quantile_embedding.py](models/quantile_embedding.py) | Sinusoidal q embedding |
| [models/quantile_model.py](models/quantile_model.py) | QUTCC/raw-Im2im wrapper, pinball objective and clean LR schedule |
| [data/paired.py](data/paired.py) | Existing paired split loader/audit, including raw parent data |
| [calibration/quantile_search.py](calibration/quantile_search.py) | Algorithm 1 quantile calibration |
| [training/paper.py](training/paper.py) | Identical clean QUTCC/raw training loop |
| [evaluation/paper.py](evaluation/paper.py) | Identical clean calibration/full-test loop |
| [evaluation/continuation.py](evaluation/continuation.py) | Shared continuation evaluator |
| [tests/paper_protocol.py](tests/paper_protocol.py) | Pinball, schedule, optimizer/RNG and calibration tests |
| [protocols/paper.json](protocols/paper.json) | Shared clean protocol settings |
| [cluster/run.sbatch](cluster/run.sbatch) | Recorded identical Slurm wrapper |

Experiment-specific baseline heads, corrupted loaders, adaptation/continuation
training and calibration choices remain under [Exp2](../experiments/experiment2/)
and [Exp3](../experiments/experiment3/). The baseline compatibility head imports
the same FiLM core rather than defining it again.

The single shared trainer uses the invoked experiment path for `ROOT`. Thus
Exp2/Exp3 outputs and source-identity checks stay separate. The experiment's
source hash reads the linked file contents, so changing shared implementation
bytes still invalidates a stale source identity. Existing historical full-state
resumes retain their original checks; use the captured source archive for an
exact historical resume. Saved state-dictionary parameter names are unchanged.

The layout uses ordinary relative Git symlinks supported by the project's macOS
and Linux environments. Preserve those links when copying the source checkout.
These shared scripts rely on the experiment import context; run the documented
experiment commands, rather than invoking a shared implementation directly.

[source_map.json](source_map.json) lists every alias and implementation hash.
[Equivalence report](../provenance/shared_code_equivalence_20261021.json) records
fixed-seed before/after CPU checks of weights, parameter keys/shapes, predictions,
losses, input/parameter gradients and RNG consumption, including nonidentity
FiLM and odd-sized raw inputs. No training jobs were restarted by this refactor.
