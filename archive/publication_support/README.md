# Optional publication support

These files are not imported by the training, evaluation, calibration or download
commands. They were moved here to keep the main repository focused on the models
and their runnable workflow.

| Folder | Contents |
| --- | --- |
| [docs/](docs/) | Extended usage, model comparison, calibration and metric guides |
| [tools/](tools/) | Learning-curve plotting, numerical equivalence and source checks |
| [workflows/](workflows/) | Optional GitHub Actions test configuration; inactive here |
| [calibration_legacy/](calibration_legacy/) | Superseded empirical modes and original source snapshots |

Run archived tools from the checkout root, for example:

```bash
python archive/publication_support/tools/plot_learning_curves.py --help
python archive/publication_support/tools/verify_source_release.py
```

To enable hosted tests in a future checkout, copy `workflows/tests.yml` to
`.github/workflows/tests.yml`. GitHub runner access and account limits still apply.
Local tests remain in [tests/](../../tests/).
