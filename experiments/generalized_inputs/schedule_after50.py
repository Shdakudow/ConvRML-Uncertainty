"""Queue epochs 51-100 after each epoch-50 job, leaving active jobs untouched."""

import argparse
from pathlib import Path

from slurm import (
    METHODS,
    read_json,
    require_same_training_code,
    submission_lock,
    submit_job,
)

SOURCE = Path(__file__).resolve().parent


def schedule_continuations(run_root, source=SOURCE):
    """Resume a partial submission safely, retaining all existing job records."""
    run_root, source = run_root.resolve(), source.resolve()
    for name in ("run.sbatch", "continue_after50.py"):
        if not (source / name).is_file():
            raise FileNotFoundError(source / name)
    protocol = read_json(source / "protocol.json")
    if (protocol["epochs"], protocol.get("base_epochs")) != (100, 50):
        raise ValueError("This scheduler requires the reviewed 50-to-100 protocol.")
    if not (source.parents[1] / "source_commit.txt").is_file():
        raise FileNotFoundError("A frozen source snapshot needs source_commit.txt.")
    with submission_lock(run_root):
        require_same_training_code(source.parents[1], run_root / "source50")
        predecessors = read_json(run_root / "jobs_50epochs.json")
        dependencies = {
            method: predecessors[f"scratch50-{method}"]["job_id"] for method in METHODS
        }
        if any(not str(job_id).isdigit() for job_id in dependencies.values()):
            raise ValueError("Each model needs a valid epoch-50 predecessor job ID.")
        record_path = run_root / "jobs_100epochs.json"
        jobs = read_json(record_path) if record_path.exists() else {}
        expected_names = {f"scratch100-{method}" for method in METHODS}
        if set(jobs) - expected_names:
            raise ValueError("Continuation records contain an unexpected job name.")
        for method in METHODS:
            submit_job(
                source=source,
                run_root=run_root,
                name=f"scratch100-{method}",
                script="continue_after50.py",
                arguments=["--run-root", str(run_root), "--method", method],
                records=jobs,
                record_path=record_path,
                dependency=dependencies[method],
            )
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    schedule_continuations(args.run_root)


if __name__ == "__main__":
    main()
