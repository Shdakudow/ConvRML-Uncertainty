# Source before simplification

`source_before_simplification.tar.gz` preserves all 109 tracked non-archive files
from revision `1da7ec3aa17f9030f4869ae3d5e62bb75c54c3d9`, byte-for-byte.
[snapshot.json](snapshot.json) records the archive SHA-256 and scope.

This includes the original training/calibration engines, tests, Slurm launchers,
mid-epoch recovery, source/software enforcement, historical epoch-extension
migrations and the robustness demo export/publishing pipeline. Dependencies,
settings and documentation are included. Dataset images and model checkpoints
are external. Existing frozen cluster and publishing deployments are unchanged.

To inspect or run the old implementation, extract into a **separate directory**:

```bash
mkdir /path/to/original-source
tar -xzf source_before_simplification.tar.gz -C /path/to/original-source
```

Use its root as the working directory and follow its README. Historical
submission tools also require the deployment's recorded `source_commit.txt`.
A Git checkout of the recorded revision is another way to obtain this source.
Do not mix archived resume/migration utilities with the simplified engine.

The current root scripts retain models, losses, numerical settings, split
isolation, finite checks and complete-epoch checkpoint/resume. They deliberately
omit the older deployment machinery and redundant package command wrappers.
The simplified generalized trainer uses its own epoch-boundary checkpoint schema;
resume historical deployments with their original engine.
