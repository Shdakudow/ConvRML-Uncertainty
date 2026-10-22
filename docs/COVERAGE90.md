# Coverage close to 90%

`tools/evaluate_coverage90.py` recalibrates **existing frozen checkpoints**, without
retraining. Its target is **90.0–90.1% calibration coverage** (a 0.1 percentage point
numerical tolerance). This is an empirical matched-coverage comparison, **not an
RCPS certificate**. Reference calibration/results remain unchanged.

- Im2im, deep ensemble and MC dropout: independent lower and upper scales are
  selected by calibration score order statistics, with 5% misses per tail. This
  avoids a coarse scale grid and RCPS confidence margins. A few floating-point
  ULPs protect selected boundary values during endpoint multiplication.
- QUTCC: refine the two **global quantile inputs**, retaining native quantile
  images. Start from the existing calibrated pair when available. Each tail is
  measured on the full calibration split and refined toward 5% misses. No extra
  interval scale or prediction head is introduced. Observed nonmonotone q responses
  fail explicitly; the method does not assume a guaranteed monotone network.
- Calibration uses **clipped [0,1] endpoints**, matching displayed metrics. Raw
  and clipped test metrics are both reported. Test targets never select parameters.
- Quantized/tied outputs can make the requested tolerance unattainable. Every
  report records actual calibration coverage and `within_tolerance`; a failed
  tolerance is never reported as exactly 90%. Test coverage is allowed to vary.

## Experiment 2

Run from the repository using the model environment and existing dataset/checkpoints:

```bash
python experiments/experiment2/run_model.py evaluate \
  --method im2im --camera rml --epoch 50 \
  --project-root /home/jy2228/ConvRML --calibration tight90

# Original cluster QUTCC result location, if not copied beside the published code:
python experiments/experiment2/run_model.py evaluate \
  --method qutcc --camera diffuser --epoch 50 \
  --project-root /home/jy2228/ConvRML --calibration tight90 \
  --calibration-source /home/jy2228/ConvRML/experiments/qutcc_paper_style_20261018
```

The baseline router defaults to the published Exp2 source folder and reads weights
under `PROJECT/results/corrected_comparison_20261006`. If those weights were moved
to a GitHub Release, restore the exact milestone first; the evaluator does not
silently substitute another model or epoch. Use `--method ensemble --members 5`
(or 10), and `--method mc_dropout --dropout 0.05`, for the other baselines.
The standalone evaluator also accepts `--weights-root` for an existing restored
baseline checkpoint tree (camera/arm/checkpoints/epoch_XXX.pt).

## Experiment 3

Select the clean or corrupted source/results folder explicitly:

```bash
python tools/evaluate_coverage90.py \
  --project-root /home/jy2228/ConvRML \
  --source-folder /home/jy2228/ConvRML/experiments/raw_corrupted_continue30_20261019 \
  --family paper --camera raw --method qutcc --epoch 30 \
  --output /home/jy2228/ConvRML/results/tight_coverage90/corrupted/qutcc/epoch_030
```

For clean raw use `qutcc_paper_style_20261018`, epoch 50. Replace `qutcc` with
`im2im` to evaluate its independent endpoint scales. Use separate output folders.

The evaluator saves `calibration.json` **before opening the test dataset**, then
`report.json` with calibration coverage, held-out test coverage/width/SSIM/PSNR/MAE,
per-image metrics, split identity, source provenance and GPU timing. `examples.npz`
contains the first 50 fixed test examples with clipped reconstruction and bounds.
Temporary calibration scores are deleted after fitting. Existing reports are
reused only for identical requests. No demo is changed automatically by this tool.
