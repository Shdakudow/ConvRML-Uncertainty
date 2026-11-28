# MIRFLICKR · QUTCC

[Download best weights (.pt)](https://github.com/Shdakudow/ConvRML-Uncertainty/releases/download/uq-weights-20261120/mirflickr-qutcc.pt) · Epoch 49 · Validation SSIM 0.81983

[Calibration parameters](calibration.pt) · [Test results and checkpoint SHA-256](results.json)

q lower = 0.041503906, q upper = 0.97680664. Coverage was 90.062% on calibration and 89.973% across 2,000 test images. Mean test interval width was 0.13018.

Run from the repository root with the downloaded checkpoint in the current directory.

```python
import torch
from models import make_model

checkpoint = torch.load("mirflickr-qutcc.pt", map_location="cpu", weights_only=True)
model = make_model(checkpoint["method"], checkpoint["height"], checkpoint["width"], checkpoint["dataset_size"])
model.load_state_dict(checkpoint["model"])
model.eval()
```

[Training archive](training-archive.md)
