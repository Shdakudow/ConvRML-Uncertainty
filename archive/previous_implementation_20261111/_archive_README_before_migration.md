# Historical material

[publication_support/](publication_support/) holds optional extended guides,
plotting/provenance tools and inactive GitHub Actions configuration.

[operations/](operations/) preserves cluster/job/demo deployment tooling. The portable release needs none of it.

The four shared model implementations are in [models/](../models/), with data
loading in [dataset.py](../dataset.py) and shared runtime/layers in [convrml/](../convrml/).

[reproduction/](reproduction/) preserves the historical numbered-experiment engines
and their original import layout, apart from the concise model interface.

| Item                                                         | Contents                                                                                                                        |
| ------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------- |
| [research_prototype_20261102/](research_prototype_20261102/) | Complete pre-simplification source, deployment engines and tests                                                                |
| [history_20261022.tar.gz](history_20261022.tar.gz)           | Earlier experiments, stopped clean continuation, obsolete job records, scheduler/recovery utilities and superseded instructions |
| [index.json](index.json)                                     | Every bundled path, size and SHA-256                                                                                            |
| [provenance/](provenance/)                                   | Original captured source, source-layout maps and numerical equivalence report                                                   |

Historical files were removed from the active tree after their exact contents
were verified in this bundle. Checkpoints, datasets and the remote cluster were
not deleted. Historical engines and imported compatibility modules remain readable under `reproduction/`.

Inspect or restore into a separate directory:

```bash
tar -tzf archive/history_20261022.tar.gz
mkdir -p /tmp/convrml-history
tar -xzf archive/history_20261022.tar.gz -C /tmp/convrml-history
```

Entries retain their former repository paths. Symlinks were stored as their
source bytes to make historical files readable independently; the index records
their original shared targets. Historical scripts may require their original
cluster paths and source snapshots. [Source verification](publication_support/tools/verify_source_release.py)
checks both this bundle and the original captured source.
