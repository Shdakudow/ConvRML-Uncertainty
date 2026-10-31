"""Queue epochs 51-100 behind each active epoch-50 job, without interrupting it."""

import argparse
import fcntl
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.run_root.resolve()
    source = Path(__file__).resolve().parent
    with (root / "schedule100.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / "jobs_100epochs.json").exists():
            raise SystemExit(
                "100-epoch continuations already scheduled; refusing duplicates."
            )
        predecessors = json.loads((root / "jobs_50epochs.json").read_text())
        current = json.loads((root / "jobs.json").read_text())
        repository = source.parents[1]
        original = root / "source50"
        paths = [Path("dataset.py")]
        for directory in ("models", "convrml", "input_modifications"):
            paths.extend(
                p.relative_to(repository)
                for p in (repository / directory).rglob("*.py")
            )
        for path in paths:
            if (repository / path).read_bytes() != (original / path).read_bytes():
                raise ValueError(f"Model/data/modification code changed: {path}")
        jobs = {}
        for method in ("qutcc", "im2im"):
            dependency = predecessors[f"scratch50-{method}"]["job_id"]
            command = [
                "/usr/local/slurm/current/bin/sbatch",
                "--parsable",
                f"--job-name=scratch100-{method}",
                f"--dependency=afterok:{dependency}",
                "--kill-on-invalid-dep=yes",
                f"--chdir={repository}",
                f"--output={root}/logs/%j.out",
                f"--error={root}/logs/%j.err",
                str(source / "run.sbatch"),
                str(source / "continue_after50.py"),
                "--run-root",
                str(root),
                "--method",
                method,
            ]
            job = subprocess.check_output(command, text=True).strip().split(";")[0]
            jobs[f"scratch100-{method}"] = dict(
                job_id=job, dependency=dependency, command=command
            )
            temporary = root / "jobs_100epochs.tmp"
            temporary.write_text(json.dumps(jobs, indent=2) + "\n")
            temporary.replace(root / "jobs_100epochs.json")
            merged = dict(current, **jobs)
            temporary = root / "jobs.tmp"
            temporary.write_text(json.dumps(merged, indent=2) + "\n")
            temporary.replace(root / "jobs.json")
            print(method, job, "after", dependency, flush=True)


if __name__ == "__main__":
    main()
