# Historical material

The active implementation is in [shared/](../shared/), [Exp2](../experiments/experiment2/) and [Exp3](../experiments/experiment3/).

| Item | Contents |
| --- | --- |
| [history_20261022.tar.gz](history_20261022.tar.gz) | Earlier experiments, stopped clean continuation, obsolete job records, scheduler/recovery utilities and superseded instructions |
| [index.json](index.json) | Every bundled path, size and SHA-256 |
| [provenance/](provenance/) | Original captured source, source-layout maps and numerical equivalence report |

Historical files were removed from the active tree after their exact contents
were verified in this bundle. Checkpoints, datasets and the remote cluster were
not deleted. Existing runtime code and its imported compatibility modules remain
in the active implementation.

Inspect or restore into a separate directory:

```bash
tar -tzf archive/history_20261022.tar.gz
mkdir -p /tmp/convrml-history
tar -xzf archive/history_20261022.tar.gz -C /tmp/convrml-history
```

Entries retain their former repository paths. Symlinks were stored as their
source bytes to make historical files readable independently; the index records
their original shared targets. Historical scripts may require their original
cluster paths and source snapshots. [Source verification](../tools/verify_source_release.py)
checks both this bundle and the original captured source.
