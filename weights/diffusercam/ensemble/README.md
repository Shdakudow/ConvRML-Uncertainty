# DiffuserCam · Deep ensemble

[Download best weights (.pt)](https://github.com/Shdakudow/ConvRML-Uncertainty/releases/download/uq-weights-20261120/diffusercam-ensemble-10members-epoch-050.pt) · Epoch 50 · Validation SSIM 0.88945

[Calibration parameters](calibration.pt) · [Test results and checkpoint SHA-256](results.json)

Shared λ = 0.62062061. Coverage was 90.129% on calibration and 90.146% across 5,000 test images. Mean test interval width was 0.03649.

Run from the repository root with the downloaded checkpoint in the current directory.

```python
import torch
from models import make_model

checkpoint = torch.load("diffusercam-ensemble-10members-epoch-050.pt", map_location="cpu", weights_only=True)
model = make_model(checkpoint["method"], checkpoint["height"], checkpoint["width"], checkpoint["dataset_size"])
model.load_state_dict(checkpoint["model"])
model.eval()
```

[Training archive](training-archive.md)
