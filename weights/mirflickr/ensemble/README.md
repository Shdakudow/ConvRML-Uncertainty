# MIRFLICKR · Deep ensemble

[Download best weights (.pt)](https://github.com/Shdakudow/ConvRML-Uncertainty/releases/download/uq-weights-20261120/mirflickr-ensemble-10members-epoch-049.pt) · Epoch 49 · Validation SSIM 0.78373

[Calibration parameters](calibration.pt) · [Test results and checkpoint SHA-256](results.json)

Shared λ = 0.89089084. Coverage was 90.272% on calibration and 90.062% across 2,000 test images. Mean test interval width was 0.13520.

Run from the repository root with the downloaded checkpoint in the current directory.

```python
import torch
from models import make_model

checkpoint = torch.load("mirflickr-ensemble-10members-epoch-049.pt", map_location="cpu", weights_only=True)
model = make_model(checkpoint["method"], checkpoint["height"], checkpoint["width"], checkpoint["dataset_size"])
model.load_state_dict(checkpoint["model"])
model.eval()
```

[Training archive](training-archive.md)
