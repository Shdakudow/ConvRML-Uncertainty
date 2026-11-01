"""Submit Gaussian-only QUTCC/Im2im specialists with matched untrained weights."""

import argparse
import hashlib
from pathlib import Path
import sys

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "experiments/generalized_inputs"))
from slurm import METHODS, read_json, submission_lock, submit_job

DEFAULT_PROJECT = Path("/home/jy2228/ConvRML")
DEFAULT_COMPARISON = (
    DEFAULT_PROJECT / "experiments/raw_online_scratch_30epochs_20261031_v2"
)


def submit_specialists(run_root, project, comparison_root, repository=REPOSITORY):
    """Match the training budget and initialization of generalized epochs 1-50."""
    repository = repository.resolve()
    run_root, project, comparison_root = (
        run_root.resolve(),
        project.resolve(),
        comparison_root.resolve(),
    )
    source = repository / "experiments/generalized_inputs"
    protocol_path = repository / "experiments/gaussian_specialist/protocol.json"
    protocol = read_json(protocol_path)
    if (
        protocol["epochs"],
        protocol["training_condition"],
        protocol["gaussian_sigma"],
    ) != (50, "gaussian", 0.01):
        raise ValueError("Expected the reviewed 50-epoch Gaussian SD 0.01 protocol.")
    manifest = project / "experiments/dlmd_raw_adaptive_size_20261016/data/split.json"
    if hashlib.sha256(manifest.read_bytes()).hexdigest() != protocol["manifest_sha256"]:
        raise ValueError("The paired MIRFLICKR manifest changed.")
    references = {
        method: comparison_root / "results" / method / "initial_random.pt"
        for method in METHODS
    }
    for path in [
        source / "run.sbatch",
        source / "train.py",
        repository / "source_commit.txt",
        *references.values(),
    ]:
        if not path.is_file():
            raise FileNotFoundError(path)
    expected_names = {
        f"gaussian50-{method}{suffix}"
        for method in METHODS
        for suffix in ("", "-smoke")
    }
    with submission_lock(run_root):
        record_path = run_root / "jobs.json"
        jobs = read_json(record_path) if record_path.exists() else {}
        if set(jobs) - expected_names:
            raise ValueError("Use a separate run directory for Gaussian specialists.")
        for method in METHODS:
            name = f"gaussian50-{method}"
            arguments = [
                "--method",
                method,
                "--protocol",
                str(protocol_path),
                "--manifest",
                str(manifest),
                "--output",
                str(run_root / "results" / method),
                "--initial-random-state",
                str(references[method]),
            ]
            smoke = submit_job(
                source=source,
                run_root=run_root,
                name=name + "-smoke",
                script="train.py",
                arguments=arguments + ["--smoke"],
                records=jobs,
                record_path=record_path,
                time_limit="00:20:00",
            )
            submit_job(
                source=source,
                run_root=run_root,
                name=name,
                script="train.py",
                arguments=arguments,
                records=jobs,
                record_path=record_path,
                dependency=smoke,
            )
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--comparison-root", type=Path, default=DEFAULT_COMPARISON)
    args = parser.parse_args()
    submit_specialists(args.run_root, args.project, args.comparison_root)


if __name__ == "__main__":
    main()
