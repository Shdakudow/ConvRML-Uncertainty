"""Continue the existing scratch runs to 50 epochs after a checkpoint handoff.

The two old jobs must already be stopped gracefully. This command preserves the
original configuration and full checkpoint before submitting each continuation.
"""

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.run_root.resolve()
    source = Path(__file__).resolve().parent
    with (root / "submit_extension.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / "jobs_50epochs.json").exists():
            raise SystemExit("The extension is already submitted; refusing duplicates.")
        old_jobs = json.loads((root / "jobs.json").read_text())
        job_ids = ",".join(row["job_id"] for row in old_jobs.values())
        running = subprocess.check_output(
            ["/usr/local/slurm/current/bin/squeue", "-h", "-j", job_ids], text=True
        )
        if running.strip():
            raise SystemExit("Stop the original jobs at a checkpoint before extending.")
        repository = source.parents[1]
        original_repository = root / "source"
        paths = [Path("dataset.py")]
        for directory in ("models", "convrml", "input_modifications"):
            paths.extend(
                path.relative_to(repository)
                for path in (repository / directory).rglob("*.py")
            )
        for path in paths:
            if (repository / path).read_bytes() != (
                original_repository / path
            ).read_bytes():
                raise ValueError(f"Model/data/modification code changed: {path}")
        (root / "jobs_30epochs.json").write_text(json.dumps(old_jobs, indent=2) + "\n")
        jobs = {}
        for method in ("qutcc", "im2im"):
            output = root / "results" / method
            previous = json.loads((output / "config.json").read_text())
            if previous["epochs"] != 30 or previous["pretrained"]:
                raise ValueError("Expected the active 30-epoch scratch configuration.")
            extension = output / "config_before_50epochs.json"
            extension.write_text(json.dumps(previous, indent=2) + "\n")
            # Atomic checkpoint replacement in the runner preserves this old inode.
            os.link(output / "latest.pt", output / "latest_before_50epochs.pt")
            command = [
                "/usr/local/slurm/current/bin/sbatch",
                "--parsable",
                f"--job-name=scratch50-{method}",
                f"--chdir={source.parents[1]}",
                f"--output={root}/logs/%j.out",
                f"--error={root}/logs/%j.err",
                str(source / "run.sbatch"),
                str(source / "train.py"),
                "--method",
                method,
                "--manifest",
                previous["manifest_path"],
                "--output",
                str(output),
                "--extension",
                str(extension),
            ]
            job = subprocess.check_output(command, text=True).strip().split(";")[0]
            jobs[f"scratch50-{method}"] = dict(job_id=job, command=command)
            temporary = root / "jobs_50epochs.tmp"
            temporary.write_text(json.dumps(jobs, indent=2) + "\n")
            temporary.replace(root / "jobs_50epochs.json")
            print(method, job, flush=True)
        temporary = root / "jobs.tmp"
        temporary.write_text(json.dumps(jobs, indent=2) + "\n")
        temporary.replace(root / "jobs.json")


if __name__ == "__main__":
    main()
