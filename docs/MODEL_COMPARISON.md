# Model-code comparison

[Exp2](../experiments/experiment2/) compares four methods on RML and normalized Cornell DiffuserCam. [Exp3](../experiments/experiment3/) compares QUTCC and Im2im on raw and fixed corrupted raw DiffuserCam.

Common implementations now live once in [shared/](../shared/). Paths linked
below remain valid experiment aliases. The [shared-code index](../shared/README.md)
links directly to the canonical files and explains which files remain specific.

## Shared reconstruction backbone

Read [`ShamusRML`](../experiments/experiment2/four_models/models/shamus_rml.py):

1. Standardize RGB input using stored ImageNet mean/std.
2. ConvNeXtV2-Base returns four feature maps with 128, 256, 512 and 1,024 channels. `pretrained=False`: training starts from random weights.
3. Four lateral 1×1 convolutions project features to 256 channels. The FPN upsamples coarse features, adds finer features, and refines them.
4. Each refinement has two 3×3 convolutions, each followed by GroupNorm and GELU. Four FPN refinements and two final upsampling/refinements give twelve decoder normalization sites.
5. A 3×3 RGB head predicts a residual, which is added to the measurement.

This is a ConvNeXt/FPN adaptation of QUTCC's quantile-conditioning method, not the paper's original attention U-Net architecture.

## Differences at a glance

| Property | Im2im | Deep ensemble | MC dropout | QUTCC |
| --- | --- | --- | --- | --- |
| Trained networks | 1 | 5 or 10 independent members | 1 | 1 |
| Output per forward | 9 channels: lower/center/upper RGB | Per member: RGB mean + RGB raw variance | RGB reconstruction | RGB quantile image at input q |
| Added modules | Enlarged final head | Separate variance head per member | 48 dropout sites | q embedding + 48 FiLM sites |
| Objective | Center MSE + two pinball terms | Gaussian NLL | MSE | Sampled-q pinball only |
| Reconstruction | Center output | Average member means | Average stochastic outputs | q=0.5 image |
| Native bounds | Predicted endpoints | Gaussian moment interval | Gaussian sample-spread interval | Two quantile-query images |
| Current passes for reconstruction + bounds | 1 | 5 or 10 | 30 | 3 |
| Runs here | Exp2 / Exp3 | Exp2 | Exp2 | Exp2 / Exp3 |

Pass counts describe the current inference implementations; QUTCC queries may also be batched.

## Im2im

**Exp2:** [`EndToEndIm2Im` / `training_loss`](../experiments/experiment2/four_models/models/corrected_uncertainty.py). **Exp3:** [`Model("im2im")`](../experiments/experiment3/clean/model.py).

The final RGB head becomes a nine-channel convolution, reshaped to `(batch, 3 heads, 3 RGB channels, height, width)`. Add the measurement to each residual. Head order is **lower, center, upper**.

```python
loss = mse(center, target)
loss += pinball(lower, target, q=0.05)
loss += pinball(upper, target, q=0.95)
```

All three terms update the entire backbone and head jointly. No pretrained backbone is frozen. The center is the reconstruction; the other heads provide native endpoints. Separate lower/upper calibration factors adjust endpoint distances without changing reconstruction.

## Deep ensemble

Read [`FormalRML("ensemble")`](../experiments/experiment2/four_models/baselines/formal_rml.py), [`DeepEnsembleMember`](../experiments/experiment2/four_models/baselines/deep_ensemble.py) and [`experiment_seeds`](../experiments/experiment2/four_models/models/corrected_uncertainty.py).

Each member keeps an RGB mean head and adds an RGB variance head. Variance is `softplus(raw_variance) + 1e-6`. Training minimizes Gaussian NLL, excluding a constant:

```python
0.5 * (variance.log() + (target - mean).square() / variance).mean()
```

Every member trains on the complete same training split with its own initialization, training RNG and dataloader shuffle seeds. They are independent fits, not disjoint data shards. Main five-member inference uses 00–04; the ten-member reference uses 00–09.

```python
prediction = mean(member_means)
variance = mean(member_variances + member_means**2) - prediction**2
```

This combines predicted noise and disagreement between members. Native bounds are `prediction ± 1.64485 * sqrt(variance)`, then independently calibrated per ensemble size. There are no quantile embeddings or FiLM layers.

## MC dropout

Read [`FullDropoutRML`](../experiments/experiment2/four_models/models/corrected_uncertainty.py) and [`dropout_inference`](../experiments/experiment2/four_models/baselines/mc_dropout.py).

- Elementwise `nn.Dropout(p)` acts on the MLP branch output before residual addition in all **36 encoder blocks**.
- Elementwise dropout follows GELU at all **12 decoder stages**.
- Training uses MSE; there is no variance head or q conditioning.
- Inference enables dropout while retaining other evaluation behavior. Thirty masked predictions supply the mean and variance (divisor T in this implementation).

Native bounds are mean ± 1.64485 × std, then receive independent lower/upper RCPS scales. Main inference uses **p=0.05, 30 passes**. The saved validation sweep compares p=0.05/0.10/0.20 at 30/60/100 passes on 5,000 validation images.

The decoder-only `FormalRML("mc")` branch remains for compatibility. Current `make_model("mc_005")` selects **`FullDropoutRML`**, not that older branch.

## QUTCC

Read [`Model("qutcc")` / `training_loss`](../experiments/experiment2/four_models/qutcc/model.py) and [`FiLMBackbone`](../experiments/experiment2/four_models/qutcc/models/convnext_norm_film.py).

```text
measurement y ──► ConvNeXt/FPN with FiLM ──► RGB residual + y ──► Q(q | y)
                           ▲
q ──► sinusoidal embedding (128) ──► MLP (512 → 512)
```

Each FiLM projection maps the 512-dimensional embedding to channel-wise scale/shift and applies `features * (1 + scale) + shift`. It follows normalization at **36 encoder blocks and 12 decoder GroupNorm sites**. Projection weights/bias start at zero, giving identity modulation initially.

Training samples one independent scalar **q ~ Uniform(0,1) per image**, shared across that image's pixels. Only pinball updates the entire network:

```python
error = target - model(measurement, q)
loss = maximum(q * error, (q - 1) * error).mean()
```

There is no extra MSE term or learned endpoint-selector branch. Reconstruction queries q=0.5. Calibration selects **one global lower/upper q pair per camera/checkpoint**; two further queries produce the bounds. Global q still gives spatially varying pixel predictions. Shared weights/FiLM do not strictly enforce monotonic quantiles.

Exp2's QUTCC and Exp3's clean model paths link to the same [implementation](../shared/models/quantile_model.py); [provenance](../experiments/experiment2/four_models/qutcc/provenance.json) records current and original hashes. Superseded adaptive-q classes are compatibility paths, not the current router's QUTCC.

## Calibration differences

Calibration freezes model weights and uses **separate calibration pairs**. It does not clip images automatically.

| Experiment/model | Fitted parameters | Procedure |
| --- | --- | --- |
| Exp2 QUTCC | Global lower/upper q | Paper Algorithm 1 adjusted empirical tail-risk search |
| Exp2 Im2im / ensemble / MC | Independent lower/upper distance scales | RCPS-HB; alpha=0.10, total delta=0.05, half budgets per tail |
| Exp3 QUTCC | Global lower/upper q | Algorithm 1, with 3,000 calibration images |
| Exp3 Im2im | Independent lower/upper distance scales | Adjusted empirical tail-risk scale search; not HB-RCPS |

For native center/endpoints `(mu, L0, U0)`, scale calibration gives:

```python
L = mu - s_lower * max(mu - L0, 1e-6)
U = mu + s_upper * max(U0 - mu, 1e-6)
```

QUTCC instead changes its queried q values. Algorithm 1's empirical per-tail budget is `(alpha - (1-alpha)/N_calibration_images)/2`, which differs from an HB confidence bound. Neither procedure promises exactly 90% coverage on each individual image.

The demo then explicitly clips prediction/bounds to [0,1]. Raw and clipped metrics are distinct. Width is `U-L`; coverage is the fraction of target RGB values inside the interval. These interval metrics alone do not evaluate an entire distribution.

## Experiment 2 versus Experiment 3

| Setting | Exp2 | Exp3 |
| --- | --- | --- |
| Measurements | RML and normalized Cornell parallel-camera DiffuserCam | Raw Lensless Mirflickr and fixed corrupted measurements |
| Methods | Four | QUTCC and Im2im |
| Shape | 256×256 | 270×480; replicate-pad right/bottom to multiples of 32, then crop |
| Train / validation / test / calibration | 60,000 / 5,000 / 5,000 / 2,048 per camera | 17,999 / 2,000 / 2,000 / 3,000 |
| QUTCC training | Float32 Adam, effective batch 4 | Same clean implementation/settings |
| Other training | BF16 AdamW baseline pipeline, batch 2 | Clean Im2im uses float32 Adam, effective batch 4 |
| Main clean training | 50 epochs | 50 epochs |
| Corruption adaptation | Not part of this comparison | 30 clean pretraining + 30 adaptation epochs |

Clean Adam uses betas (0.9,0.99), no weight decay, 30% warmup from 1e-5 to 1e-4, then cosine decay to 1e-9 over 50 epochs. Corruption adaptation uses constant 1e-5; the first ten adaptation epochs reset the optimizer, while the continuation retains its state through epoch 30.

This is not a strictly identical-optimizer ablation: uncertainty mechanisms, objectives and some precision/optimization settings differ. Report those settings with results. Within each experiment, methods share the same dataset split IDs.

Exp2 and Exp3 Im2im implement the same center/endpoints design in different
wrappers (`base.*` versus `net.*` state-dictionary names). Structural similarity
does not make their checkpoint files directly interchangeable. Each evaluator
loads the method/camera checkpoint and dataset identities recorded for its run.
