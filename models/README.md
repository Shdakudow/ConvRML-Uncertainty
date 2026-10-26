# Four shared model implementations

Models are defined once. Im2im and QUTCC are shared by the Parallel Lensless
Dataset (PLD, ConvRML 100k) and the DiffuserCam Lensless Mirflickr Dataset (DLMD).
Ensemble and MC-dropout comparisons currently use PLD. Dataset-specific loading
and optimization settings are in [dataset.py](../dataset.py) and
[configs/](../configs/), not in separate model branches.

| Method | Implementation | Training objective | Uncertainty |
| --- | --- | --- | --- |
| Im2im | [im2im.py](im2im.py) | Center MSE + fixed endpoint pinball, end to end | Lower/upper output heads; RCPS after training |
| QUTCC | [qutcc.py](qutcc.py) | Uniform sampled-q pinball only | Quantile embedding/FiLM; calibrated q inputs |
| Deep ensemble | [ensemble.py](ensemble.py) | Gaussian NLL for each independent member | Predicted variance and member disagreement; RCPS |
| MC dropout | [mc_dropout.py](mc_dropout.py) | MSE with dropout at 48 sites | 30 stochastic passes, p=0.05; RCPS |

Each model file contains its loss. All methods use the shared ConvNeXtV2-Base/FPN
backbone in [convrml/](../convrml/). Existing PLD runs cover all four methods;
existing DLMD runs cover Im2im and QUTCC.

```python
from models import Im2im, QUTCC

im2im = Im2im()
qutcc = QUTCC()
```

Im2im and QUTCC accept both prepared image sizes. They pad to multiples of 32
and crop every output back to the original dimensions before computing losses.

For original checkpoint compatibility, Im2im accepts `checkpoint_prefix="base"`
(PLD) or `"net"` (DLMD). This preserves historical state-dictionary names only;
both options use the same class, layers, forward pass and loss. The common
checkpoint loader chooses it automatically. No dataset-specific model file is
needed. Trained weights still belong to their own acquisition/dataset.
