# Plotting and provenance tools

| File | Purpose |
| --- | --- |
| [plot_learning_curves.py](plot_learning_curves.py) | Plot five-epoch SSIM/PSNR/width/coverage plus LPIPS/Pearson/Spearman/MAE when present |
| [check_shared_equivalence.py](check_shared_equivalence.py) | Developer check of fixed-seed model state, forward/backward and RNG |
| [verify_source_release.py](verify_source_release.py) | Optional verification of preserved original source archives |

The portable workflow uses `python -m convrml.train`, `convrml.evaluate` and
`convrml.download`; it does not import these provenance tools. Cluster dispatch,
job recovery and demo exporters are preserved in [archive/operations](../../operations/).
