"""Raw DiffuserCam QUTCC; the architecture matches the parallel lensless models.

Data: paired RGB image files via ``convrml.data.ImagePairs``; targets stay fixed
for corrupted measurements. Normalize uint8 RGB by 255. Input/output shape is
(N, 3, 270, 480). The shared model pads to 288 x 480 and crops back internally.
Only data, split sizes and training protocol differ; no extra model layers.
"""

from experiments.parallel_lensless.models.qutcc import QUTCC

__all__ = ["QUTCC"]
