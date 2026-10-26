# Deployment-specific operations

These scripts preserve the original cluster dispatch, checkpoint restoration,
legacy evaluation and demo-export workflows. They retain the original deployment
paths and scheduler assumptions and are not imported by the portable package.
Active deployed copies and queued jobs remain separate from this repository.

Use [the public commands](../../docs/RUNNING_EXPERIMENTS.md) for new reproducible
training/evaluation. Historical tests are under [tests/](tests/); the original
coverage-deployment instructions remain in [COVERAGE90.md](COVERAGE90.md).
