# Exact source and formatted publication

The repository is now organized into `experiments/experiment2/` and
`experiments/experiment3/`. `layout_migration_20261021.json` maps former source
locations to current folders, including moves to `archive/`. The archive below
keeps its original member names and hash; `tools/verify_source_release.py`
resolves the migration before comparing the published code. Model source bytes,
checkpoint parameter names and historical cluster data/output paths are preserved.

`unicorn_source_20261020.tar.gz` contains the source captured from `/home/jy2228/ConvRML` on October 20, 2026. It includes the current QUTCC/raw studies, analysis/inference exports, and baseline scripts with their imported project dependencies. It excludes datasets, credentials, weights, logs and prediction arrays.

`unicorn_source_20261020.json` lists every captured file's original SHA-256 and the archive hash. Readable copies in `experiments/` are formatted with Black; their Python ASTs are checked against this archive by `tools/verify_source_release.py`. The older October-19 manifest describes its earlier unformatted snapshot, not the formatted files' current byte hashes.

Historical training checkpoints store source/config/data identities. Formatting changes source byte hashes even when it preserves behavior. **Do not weaken those checks** to resume an old experiment. Restore exact historical source in a separate directory and use the correct full-state checkpoint and unchanged data paths. Compact `epoch_NNN.pt` snapshots contain weights/history but not the optimizer/RNG state required for exact continuation.

Extract a source copy for inspection:

```bash
mkdir -p /tmp/convrml-source-20261020
tar -xzf provenance/unicorn_source_20261020.tar.gz -C /tmp/convrml-source-20261020
```

To resume on Unicorn, restore the exact experiment subtree to a separate checkout with the original dataset identities and required parent checkpoint paths, then compare its source hash against the saved checkpoint configuration. Do not extract over an active training directory. The archive records source files, not datasets or a fully relocatable training environment.

New training may use the formatted files with a fresh smoke report and new outputs. Frozen inference can read existing state dictionaries because formatting does not change parameter keys or tensor shapes.
