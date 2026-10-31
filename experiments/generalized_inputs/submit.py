"""Submit two smoke-gated, resumable 30-epoch jobs on Unicorn."""

import argparse
import fcntl
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--project", type=Path, default=Path("/home/jy2228/ConvRML"))
    args = parser.parse_args()
    root = args.run_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir(exist_ok=True)
    with (root / "submit.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / "jobs.json").exists():
            raise SystemExit("Existing job record: refusing duplicate submission.")
        source = Path(__file__).resolve().parent
        manifest = (
            args.project / "experiments/dlmd_raw_adaptive_size_20261016/data/split.json"
        )
        parents = args.project / "experiments/qutcc_paper_style_20261018/results/raw"
        jobs = {}

        def submit(label, arguments, dependency=None, smoke=False):
            command = [
                "/usr/local/slurm/current/bin/sbatch",
                "--parsable",
                f"--job-name={label}",
                f"--chdir={source.parents[1]}",
                f"--output={root}/logs/%j.out",
                f"--error={root}/logs/%j.err",
            ]
            if smoke:
                command.append("--time=00:20:00")
            if dependency:
                command.append(f"--dependency=afterok:{dependency}")
                command.append("--kill-on-invalid-dep=yes")
            command += [
                str(source / "run.sbatch"),
                str(source / "train.py"),
            ] + arguments
            job = subprocess.check_output(command, text=True).strip().split(";")[0]
            jobs[label] = dict(job_id=job, command=command)
            temporary = root / "jobs.tmp"
            temporary.write_text(json.dumps(jobs, indent=2) + "\n")
            temporary.replace(root / "jobs.json")
            print(label, job, flush=True)
            return job

        for method in ("qutcc", "im2im"):
            arguments = [
                "--method",
                method,
                "--manifest",
                str(manifest),
                "--initialize",
                str(parents / method / "checkpoints/epoch_050.pt"),
                "--output",
                str(root / "results" / method),
            ]
            smoke = submit(
                f"online30-{method}-smoke", arguments + ["--smoke"], smoke=True
            )
            submit(f"online30-{method}", arguments, dependency=smoke)


if __name__ == "__main__":
    main()
