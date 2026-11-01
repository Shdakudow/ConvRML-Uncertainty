"""Submit one smoke test and one dependent scratch-training job per model."""

import argparse
import hashlib
from pathlib import Path

from slurm import METHODS, read_json, submission_lock, submit_job

SOURCE = Path(__file__).resolve().parent
DEFAULT_PROJECT = Path("/home/jy2228/ConvRML")


def submit_training(run_root, project, source=SOURCE):
    """Validate the inputs, then submit or reuse the four jobs in the plan."""
    run_root, project, source = run_root.resolve(), project.resolve(), source.resolve()
    manifest = project / "experiments/dlmd_raw_adaptive_size_20261016/data/split.json"
    if not manifest.is_file():
        raise FileNotFoundError(f"Training manifest does not exist: {manifest}")
    for name in ("run.sbatch", "train.py"):
        if not (source / name).is_file():
            raise FileNotFoundError(source / name)
    protocol = read_json(source / "protocol.json")
    if hashlib.sha256(manifest.read_bytes()).hexdigest() != protocol["manifest_sha256"]:
        raise ValueError("Training manifest does not match the recorded protocol.")
    if not (source.parents[1] / "source_commit.txt").is_file():
        raise FileNotFoundError("A frozen source snapshot needs source_commit.txt.")
    epochs = protocol["epochs"]
    expected_names = {
        f"scratch{epochs}-{method}{suffix}"
        for method in METHODS
        for suffix in ("", "-smoke")
    }
    with submission_lock(run_root):
        record_path = run_root / "jobs.json"
        jobs = read_json(record_path) if record_path.exists() else {}
        if set(jobs) - expected_names:
            raise ValueError(
                "This folder contains a different submission plan; use a new run folder."
            )
        for method in METHODS:
            name = f"scratch{epochs}-{method}"
            arguments = [
                "--method",
                method,
                "--manifest",
                str(manifest),
                "--output",
                str(run_root / "results" / method),
            ]
            smoke_id = submit_job(
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
                dependency=smoke_id,
            )
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    args = parser.parse_args()
    submit_training(args.run_root, args.project)


if __name__ == "__main__":
    main()
