# Common source

Each implementation lives once here. Relative links from experiment folders
retain their import context and separate outputs. Run the documented experiment
entry points, rather than these shared engines directly.

| Folder | Implementation |
| --- | --- |
| [models/](models/) | ConvNeXt/FPN backbone, FiLM, q embedding and quantile/raw-Im2im wrapper |
| [data/](data/) | Existing paired split loader and audit |
| [calibration/](calibration/) | Algorithm 1 quantile search |
| [training/](training/) | Shared clean training loop |
| [evaluation/](evaluation/) | Clean and continuation calibration/test loops |
| [protocols/](protocols/) | Common clean settings |
| [tests/](tests/) | Shared pinball, schedule, optimizer/RNG and calibration tests |

[source_map.json](source_map.json) lists active aliases and source hashes.
[Verification evidence](../archive/provenance/shared_code_equivalence_20261021.json)
records exact fixed-seed CPU matches before/after sharing. Historical source
and cluster wrappers are in the [archive](../archive/).

Model parameter names and settings are preserved. Exact historical full-state
resume still requires its recorded source/data identities; those checks remain
active. Preserve relative Git symlinks when copying this macOS/Linux checkout.
