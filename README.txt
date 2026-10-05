ConvRML uncertainty experiments — source snapshot, 2026-10-05
==========================================================
Current study: selective FiLM, 50 total epochs, RML and DiffuserCam.

Parameters: configs/selective_film50_parameters.json
Training: scripts/selective_film50_study.py
Submission: scripts/submit_selective_film50.py
Slurm template: scripts/selective_film50_study.sbatch
Evaluation and validation selection: scripts/selective_film50_evaluate.py
FiLM implementation: models/convnext_norm_film.py
Baseline architecture: models/shamus_rml.py and baselines/formal_rml.py

Im2im is NOT retrained. Its existing epoch-50 MSE backbone and trained fixed
endpoint head are reused. The original five ensemble members are also reused.
Five additional ensemble members and three MC-dropout variants resume from
pilot states; only the new QUTCC architecture starts fresh. Only QUTCC has FiLM.

The current study uses MSE, Gaussian NLL and pinball as documented in the
parameters JSON. It is not the historical CRPS toy study. Older experiment
scripts are retained for provenance; the selective_film50 entry points above
are the current workflow. Never run every submission script indiscriminately.

Runtime: Python 3.11, PyTorch/CUDA appropriate for the GPU, plus requirements.txt.
Run commands from this directory. SLURM scripts currently expect ~/ConvRML;
copy this experiment directory there or edit the working directory first.
The preserved pilot checkpoints and existing 50-epoch baselines are required
for this resume workflow. Dataset caches, private split manifests, model weights
and machine-specific runtime logs are not embedded in this source tree.

Data caches: data/optics_intervals_60k_5k_5k_20260928 and
 data/diffuser_intervals_60k_5k_5k_20260928 (pairs.npy + manifest.json).
Separate calibration caches: data/formal_rml_calibration95 and
 data/diffuser_intervals_calibration_20260928.
The fixed split is 60,000 train, 5,000 validation, 5,000 test, with 2,048
separate calibration images. Do not recreate splits independently for methods.

Machine-specific cleanup and automatic demo-publishing scripts are excluded.
Job IDs and cluster paths are not portable. Review submission scripts before
allocating compute. This repository does not deploy or modify the demo.

Checks:
 python scripts/test_convnext_norm_film.py
 python -m compileall -q models baselines scripts

Sources:
 https://github.com/lakabuli/ConvRML (upstream; LICENSE and CITATION.cff retained)
 https://github.com/Shdakudow/code-for-quant (original QUTCC implementation;
  provenance in models/qutcc_original/provenance.json)
Demo:
 https://github.com/Shdakudow/ConvRML-Uncertainty-Demo
 https://shdakudow.github.io/ConvRML-Uncertainty-Demo/
The demo currently represents earlier completed runs, not the new FiLM study.
