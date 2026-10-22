# Experiment 3 · raw and corrupted raw DiffuserCam

Compare **QUTCC and end-to-end Im2im** on raw DiffuserCam Lensless Mirflickr. Images are RGB 270×480; the model pads to a multiple of 32 internally and crops outputs back to the original dimensions.

| Folder | Role |
| --- | --- |
| **[clean/](clean/)** | Clean raw training from random initialization, 50 epochs |
| **[corrupted/](corrupted/)** | Fixed reviewed sensor corruptions and first ten adaptation epochs, initialized from clean epoch 30 |
| **[corrupted_continuation/](corrupted_continuation/)** | Retains adaptation optimizer/RNG through 30 adaptation epochs |
| [exports/](exports/) | Frozen-checkpoint image/bound exports |
| [clean_continuation_stopped/](clean_continuation_stopped/) | Exp3 follow-up stopped at epoch 70; historical schedule retained |

[Code differences](../../docs/MODEL_COMPARISON.md#experiment-2-versus-experiment-3) · [commands](../../docs/RUNNING_EXPERIMENTS.md#experiment-3)

## Changes from Experiment 2

- Only QUTCC and Im2im are trained here.
- The raw-compatible [`Model`](clean/model.py) wraps both methods. QUTCC's 48 FiLM sites and sampled-q pinball objective match Exp2's current QUTCC. Im2im retains the same nine-channel center/endpoints design, with pad/crop handling.
- Both clean raw models use float32 Adam, effective batch four and a 50-epoch warmup/cosine schedule. Exp2's three non-QUTCC baselines use their BF16/AdamW pipeline.
- Separate calibration searches global q for QUTCC and fits asymmetric scales for Im2im using adjusted empirical tail risk. Exp3 Im2im does not use Exp2's RCPS-HB procedure.

The split is **17,999 train / 2,000 validation / 2,000 test / 3,000 calibration**. Both models share the paired split; the audit found 17,999 usable training pairs from the original 18,000 request.

## Corrupted raw inputs

The manifest assigns one fixed operation per image: identity, crop/resize, rotation, local patch rotation, intensity change, missing pixels, reduced resolution, Gaussian noise or Poisson noise. The mixture is across images; individual inputs are **not hybrids**. Targets and split IDs stay unchanged. Read [`corruptions.py`](corrupted/corruptions.py) and [`build_dataset.py`](corrupted/build_dataset.py).

The first ten adaptation epochs start from clean epoch-30 weights with a fresh optimizer. The continuation retains its state through adaptation epoch 30 at learning rate 1e-5. Final corrupted checkpoints represent **30 clean + 30 adaptation epochs**.

Common code lives once in [shared/](../../shared/); the experiment filenames are relative links. Corruption/adaptation-specific files remain in their own folders. The clean entry point retains historical normalized-camera task IDs 0/1 alongside raw tasks 2/3. Use **2 = raw QUTCC** and **3 = raw Im2im** here. The original pre-refactor source remains in the provenance archive.
