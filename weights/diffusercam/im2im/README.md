# DiffuserCam · Im2im

[Download best weights (.pt)](https://github.com/Shdakudow/ConvRML-Uncertainty/releases/download/uq-weights-20261120/diffusercam-im2im-epoch-049.pt) · Epoch 49 · Validation SSIM 0.92332

[Calibration parameters](calibration.pt) · [Test results and checkpoint SHA-256](results.json)

Spatial offset (min 0, mean 0, max 0). Coverage was 93.728% on calibration and 93.730% across 5,000 test images. Mean test interval width was 0.03365.

Run from the repository root with the downloaded checkpoint in the current directory.

```python
import torch
from models import make_model

checkpoint = torch.load("diffusercam-im2im-epoch-049.pt", map_location="cpu", weights_only=True)
model = make_model(checkpoint["method"], checkpoint["height"], checkpoint["width"], checkpoint["dataset_size"])
model.load_state_dict(checkpoint["model"])
model.eval()
```
