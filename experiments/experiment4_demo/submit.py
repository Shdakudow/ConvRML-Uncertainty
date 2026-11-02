"""Submit one resumable Exp 4 evaluation after the existing QUTCC training job."""

import argparse
import fcntl
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
SLURM = Path("/usr/local/slurm/current/bin")
PROJECT = Path("/home/jy2228/ConvRML/experiments")


def main(args):
    args.root.mkdir(parents=True, exist_ok=True)
    (args.root / "logs").mkdir(exist_ok=True)
    with (args.root / "submission.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        command = [
            str(SLURM / "sbatch"),
            "--parsable",
            "--account=monakhova",
            "--job-name=exp4-calibration",
            f"--dependency=afterok:{args.after}",
            f"--chdir={HERE.parents[1]}",
            f"--output={args.root}/logs/%j.out",
            f"--error={args.root}/logs/%j.err",
            str(HERE / "run.sbatch"),
            str(HERE / "evaluate.py"),
            "--manifest",
            str(PROJECT / "dlmd_raw_adaptive_size_20261016/data/split.json"),
            "--generalized-root",
            str(PROJECT / "raw_online_scratch_30epochs_20261031_v2"),
            "--clean-root",
            str(PROJECT / "qutcc_paper_style_20261018"),
            "--baseline-reference",
            str(args.root / "baseline_reference.json"),
            "--output",
            str(args.root / "output"),
        ]
        record = args.root / "submission.json"
        if record.exists():
            old = json.loads(record.read_text())
            if old["command"] != command:
                raise ValueError("Recorded submission differs; refusing a duplicate")
            print(json.dumps(old))
            return
        if not (args.root / "baseline_reference.json").is_file():
            raise ValueError("Missing original demo parity reference")
        dependency = subprocess.check_output(
            [str(SLURM / "scontrol"), "show", "job", str(args.after)], text=True
        )
        if (
            "JobName=scratch100-qutcc" not in dependency
            or "UserId=jy2228(" not in dependency
        ):
            raise ValueError("Dependency is not the existing QUTCC 100-epoch job")
        result = subprocess.check_output(command, text=True).strip()
        job = result.split(";")[0]
        if not job.isdigit():
            raise ValueError("Unrecognized Slurm response")
        row = dict(job=job, dependency=args.after, command=command)
        temporary = record.with_suffix(".tmp")
        temporary.write_text(json.dumps(row, indent=2) + "\n")
        temporary.replace(record)
        print(json.dumps(row))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--after", type=int, required=True)
    main(parser.parse_args())
