# MIRFLICKR · Im2im

[Download best weights (.pt)](https://github.com/Shdakudow/ConvRML-Uncertainty/releases/download/uq-weights-20261120/mirflickr-im2im.pt) · Epoch 50 · Validation SSIM 0.82044

[Calibration parameters](calibration.pt) · [Test results and checkpoint SHA-256](results.json)

Spatial offset (min 0, mean 0.00895018, max 0.0179004). Coverage was 91.070% on calibration and 90.895% across 2,000 test images. Mean test interval width was 0.11386.

Run from the repository root with the downloaded checkpoint in the current directory.

```python
import torch
from models import make_model

checkpoint = torch.load("mirflickr-im2im.pt", map_location="cpu", weights_only=True)
model = make_model(checkpoint["method"], checkpoint["height"], checkpoint["width"], checkpoint["dataset_size"])
model.load_state_dict(checkpoint["model"])
model.eval()
```

[Training archive](training-archive.md)
