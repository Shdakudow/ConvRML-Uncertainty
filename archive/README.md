# Archive (not needed)

[Historical model weights](weights/) covers runs before November 15, 2026.

The active implementation is at the repository root in `models/`, `ConvNeXt/`,
`calibration/`, `dataloader.py`, and `train.py`.

[`previous_implementation_20261111/`](previous_implementation_20261111/)
preserves the implementation replaced on 2026-11-11, from commit
`f0b6c1f13523e154fcf8ba5de765807ef1306399`. Its original directory layout and file contents are
retained. `SNAPSHOT.json` maps every tracked file from that commit to its
preserved location and records its SHA-256. The previous archive guide is
retained there as `_archive_README_before_migration.md`.

Earlier historical material remains in:

- [`reproduction/`](reproduction/): numbered experiment implementations.
- [`research_prototype_20261102/`](research_prototype_20261102/): earlier source snapshot.
- [`operations/`](operations/): cluster and deployment tooling.
- [`publication_support/`](publication_support/): earlier guides and verification tools.
- [`reproducibility/`](reproducibility/): earlier verification records.
- [`provenance/`](provenance/): original source and attribution records.
- [`checkpoints/`](checkpoints/): checkpoint inventory.
- `history_20261022.tar.gz` and `index.json`: the earlier bundled archive.

Archived files are historical references and are excluded from active checks.
They may depend on their original environment and paths. Dataset files,
checkpoints, running cluster jobs, and the demo are outside this migration.
