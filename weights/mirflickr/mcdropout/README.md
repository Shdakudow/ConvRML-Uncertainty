# MIRFLICKR · MC dropout

[Download best weights (.pt)](https://github.com/Shdakudow/ConvRML-Uncertainty/releases/download/uq-weights-20261120/mirflickr-mcdropout-selected-50pass.pt) · Epoch 44 · Validation SSIM 0.81531

[Calibration parameters](calibration.pt) · [Test results and checkpoint SHA-256](results.json)

Shared λ = 3.0015008. Coverage was 90.056% on calibration and 90.076% across 2,000 test images. Mean test interval width was 0.10788.

Run from the repository root with the downloaded checkpoint in the current directory.

```python
import torch
from models import make_model

checkpoint = torch.load("mirflickr-mcdropout-selected-50pass.pt", map_location="cpu", weights_only=True)
model = make_model(checkpoint["method"], checkpoint["height"], checkpoint["width"], checkpoint["dataset_size"])
model.load_state_dict(checkpoint["model"])
model.eval()
```
