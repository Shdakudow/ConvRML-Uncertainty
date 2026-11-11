# Data availability and prepared formats

Image data are external to this source release. Obtain the original datasets from
their providers:

- [Parallel Lensless Dataset (PLD, ConvRML 100k)](https://lakabuli.github.io/ConvRML/):
  RML and diffuser measurements with corresponding ground truth, provided through
  [the ConvRML repository](https://github.com/lakabuli/ConvRML).
- [DiffuserCam Lensless Mirflickr Dataset (DLMD, 25k)](https://waller-lab.github.io/LenslessLearning/dataset.html):
  the separate aligned DiffuserCam/lensed-image collection used by `raw_clean`
  and its fixed-corruption derivative, `raw_corrupted`.

The PLD diffuser and DLMD are different physical measurement collections. Use
the correct acquisition and prepared preprocessing for each
checkpoint. This release supplies the saved split IDs and loader schemas; it does
not redistribute the image collections or silently replace preprocessing with a
new resize/normalization recipe.

## Prepared input layout

```text
prepared-data/
  rml/             pairs.npy, manifest.json
    calibration/   pairs.npy, manifest.json
  diffuser/        pairs.npy, manifest.json
    calibration/   pairs.npy, manifest.json
  raw_clean/       split.json, measurement/target image files
  raw_corrupted/   split.json, fixed corrupted measurements and unchanged targets
```

[Loader schemas](../../convrml/README.md) document every field; implementations are
in [dataset.py](../../dataset.py). NumPy caches are
uint8 `(N,2,3,256,256)` in train/validation/test order, with a separate calibration
cache. Image manifests contain paired RGB paths and separate roles. Both loaders
return float32 RGB divided by 255; raw input files here are **prepared RGB images**,
not unconverted 12-/16-bit sensor tensors. Preserve the original calibration,
registration, crop and intensity preprocessing used to create those pairs.

Relative image paths resolve against the manifest directory. Absolute paths are
also accepted for existing datasets. Targets and ordered IDs must remain unchanged
for fixed-corruption data. The [published split indexes](splits) let you verify
membership/order without transferring images; they are not paired-path manifests.
Formal training/evaluation checks the recorded counts and ID hashes. Small
synthetic fixtures are allowed only with an explicitly marked smoke run or in tests.

If you only have original sensor files, first prepare the matching RGB pairs using
the original dataset/preprocessing pipeline. Feeding native sensor-bit-depth files
through an 8-bit RGB loader is not an equivalent experiment.
