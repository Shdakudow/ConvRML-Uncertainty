# DiffuserCam · QUTCC

[Download best weights (.pt)](https://github.com/Shdakudow/ConvRML-Uncertainty/releases/download/uq-weights-20261120/diffusercam-qutcc-epoch-048.pt) · Epoch 48 · Validation SSIM 0.92256

[Calibration parameters](calibration.pt) · [Test results and checkpoint SHA-256](results.json)

q lower = 0.098632812, q upper = 0.90820312. Coverage was 90.053% on calibration and 90.080% across 5,000 test images. Mean test interval width was 0.02764.

Run from the repository root with the downloaded checkpoint in the current directory.

```python
import torch
from models import make_model

checkpoint = torch.load("diffusercam-qutcc-epoch-048.pt", map_location="cpu", weights_only=True)
model = make_model(checkpoint["method"], checkpoint["height"], checkpoint["width"], checkpoint["dataset_size"])
model.load_state_dict(checkpoint["model"])
model.eval()
```
