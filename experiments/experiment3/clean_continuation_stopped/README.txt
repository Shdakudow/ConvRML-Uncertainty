Original raw DiffuserCam: epoch 50 to epoch 80 continuation

Both QUTCC and Im2im resume their own completed clean-data epoch-50 latest.pt.
Model weights, Adam moments, quantile RNG, CPU/CUDA RNG, data identities and
training counters are checked and retained. The source weights are read-only.
The reviewed corruption experiment is separate; its 10-to-30 adaptation continuation
is documented in ../raw_corrupted_continue30_20261019/README.md.

Dataset: 17,999 training / 2,000 validation / 2,000 test / 3,000 calibration.
Architecture and objectives are unchanged from the paper-style experiment.
QUTCC: uniform scalar q per image, 48 FiLM sites, pinball only.
Im2im: end-to-end center MSE plus lower/upper pinball losses.

The original cosine schedule ended near 1e-9 at epoch 50. The additional phase
restarts conservatively at 1e-5 and cosine-decays to 1e-9 over 30 more epochs.
This is an explicit continuation schedule, not a retroactive change to epochs 1-50.
Adam moments are not reset. Both methods use the same continuation schedule.

Each method has one GPU smoke test and six five-epoch training jobs.
Checkpoints and separate calibration/full-test evaluation are saved at
55, 60, 65, 70, 75 and 80. Full resumable checkpoints are also saved every
500 optimizer updates and on graceful preemption.
