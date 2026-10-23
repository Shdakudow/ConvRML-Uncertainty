# Tools

| Tool | Purpose |
| --- | --- |
| [plot_learning_curves.py](plot_learning_curves.py) | CSV/PNG curves from completed reports: PSNR, SSIM, width and coverage |
| [evaluate_coverage90.py](evaluate_coverage90.py) | Frozen-checkpoint empirical calibration close to 90%; [instructions](../docs/COVERAGE90.md) |
| [run_coverage90_milestone.py](run_coverage90_milestone.py) | Run one saved checkpoint and record structured failure states |
| [dispatch_coverage90_milestones.py](dispatch_coverage90_milestones.py) | Restore one baseline epoch at a time and advance the evaluation queue |
| [coverage90_cluster.py](coverage90_cluster.py) | Cluster operations and shared report validation; accepts JSON input |
| [export/](export/) | Frozen-checkpoint prediction/bound export helpers |
| [verify_source_release.py](verify_source_release.py) | Verify historical bundles and preserved source algorithms |
| [check_shared_equivalence.py](check_shared_equivalence.py) | Fixed-seed before/after checks of weights, predictions, gradients and RNG |

See [run commands](../docs/RUNNING_EXPERIMENTS.md). Historical cluster scheduling
and publishing scripts are in the [archive](../archive/).
