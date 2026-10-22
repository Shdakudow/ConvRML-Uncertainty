# Experiment 3 · raw and corrupted raw DiffuserCam

QUTCC and end-to-end Im2im on RGB 270×480 measurements. Split:
**17,999 train / 2,000 validation / 2,000 test / 3,000 separate calibration pairs**.

| Folder | Role |
| --- | --- |
| **[clean/](clean/)** | Clean raw training from scratch, 50 epochs |
| **[corrupted/](corrupted/)** | Reviewed fixed corruption dataset and first ten adaptation epochs |
| **[corrupted_continuation/](corrupted_continuation/)** | Continues adaptation to epoch 30 with optimizer/RNG retained |

Common model/pipeline implementations live once in [shared/](../../shared/).
Raw inputs are padded internally to a multiple of 32, then outputs are cropped
back. QUTCC uses pinball-only q conditioning; Im2im jointly learns center and
endpoints. Clean raw tasks: **2 QUTCC / 3 Im2im**. Adaptation tasks: **0 QUTCC /
1 Im2im**.

Corruptions change measurements, with unchanged targets. Each image receives
one fixed operation: identity, crop/resize, rotation, local patch rotation,
intensity, missing pixels, reduced resolution, Gaussian or Poisson noise.
Final adapted models have **30 clean + 30 adaptation epochs**.

[Run instructions](../../docs/RUNNING_EXPERIMENTS.md#experiment-3) ·
[model/calibration differences](../../docs/MODEL_COMPARISON.md) ·
[prediction exports](../../tools/export/) · [stopped follow-ups and run history](../../archive/).
