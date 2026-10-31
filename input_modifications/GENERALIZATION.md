# Training a model across raw-input modifications

The [30-epoch training runner](../experiments/generalized_inputs/README.md)
implements the agreed online mixture with fresh random initialization.
Training progress and finished results must be distinguished from the earlier
frozen-model inference pilots; those pilots did not train generalized weights.

## What two models means

One generalized QUTCC and one generalized Im2im are a reasonable first comparison.
Each architecture must see the full training mixture. Two models trained on only
two fixed modification settings do not cover all the other conditions. Separately
trained specialists are useful baselines or optional distillation teachers; their
weights do not automatically combine into a generalized model.

Train each generalized model from scratch with no pretrained or clean-checkpoint
weights, while retaining the architecture's established objective. Preserve
QUTCC's sampled-quantile pinball objective and Im2im's center
and endpoint objectives. Keep a fixed, held-out validation panel per condition to
measure tradeoffs against the clean checkpoint and any trained specialists.

## Sample types and strengths during training

Generate modifications from original **training measurements only**, retaining
the paired original scene targets. Sample a type and its parameters anew as
training proceeds, rather than repeatedly seeing one fixed corrupted copy.

The agreed starting mixture is 10% clean inputs and 10% for each of the nine
method families. [online.py](online.py) implements this online, without
saving modified datasets. Exposure's brighter/darker settings form one family,
with 5% probability each overall. These frequencies apply in expectation, not as
exact batch quotas. Strengths stay at the settings below for this first version;
sampling multiple strengths is a possible later experiment, not implemented here.
The mixture is an experimental starting point, not an established optimum.

The current endpoint settings are Gaussian SD 0.01, effective Poisson count scale
1330, global exposure gains 0.50/1.50, 10% dead locations, fixed-pattern gain SD
0.01/offset SD 0.003, blur sigma 0.5, resolution factor 64, and JPEG quality 50 with
4:4:4 subsampling. These endpoints are experimental controls, not sufficient
evidence that every intermediate strength or combination will work well.

Also vary defect masks and gain/offset patterns across simulated sensors. Keep a
sensor's pattern fixed within its frames or clip. A single persistent mask across
the whole training collection can teach recovery of that mask without establishing
robustness to a different defective sensor. Hold out sensor patterns for evaluation.

Start with individual modification families. Evaluate combinations separately;
if combined degradations matter in deployment, add physically ordered combinations
to training. Arbitrarily mixing transforms or reversing their order changes the
measurement model. Do not use held-out test images to select strengths, sampling
probabilities, stopping epochs, or calibration parameters.

## Known versus unknown parameters

If the parameters are available at deployment, a conditional model can receive a
method encoding plus normalized parameters, for example:

```text
QUTCC: prediction = model(measurement, quantile_q, degradation_condition)
Im2im: center_and_endpoints = model(measurement, degradation_condition)
condition = [method encoding, Gaussian SD, log photon scale,
             log global gain, defect fraction, fixed-pattern amplitudes,
             blur sigma, log2 resolution factor, JPEG quality / 100]
```

Use explicit active-parameter indicators so unused fields cannot be confused with
physical zero values. A parameter encoder can supply feature-conditioning values;
QUTCC's quantile q remains a separate semantic input. This conditioning path is
not implemented by the current modification package.

If parameters will be unknown at deployment, train a blind model or an estimated
degradation encoder, and evaluate it without ground-truth modification labels.
An oracle-conditioned experiment is not a demonstration of blind robustness.
Parameter conditioning is an established restoration pattern, illustrated for
noise levels by [FFDNet](https://arxiv.org/abs/1710.04026); applying it to this
lensless reconstruction system remains an experiment to validate.

## Validation and uncertainty

Report clean performance, each modification, unseen strengths, new sensor patterns,
and relevant combinations separately. Compare reconstruction PSNR/SSIM/LPIPS and
interval coverage/width; a single average can hide a failed condition. Strong
information loss can limit what any model can reconstruct, even after training.

After freezing the jointly trained checkpoint, calibrate on a separate mixture
representative of intended deployment. Overall near-90% empirical coverage does
not establish near-90% coverage for every condition. Check group coverage and
width explicitly, and use group-specific or conditional calibration only when its
required condition information is available. Keep the final test split separate.
