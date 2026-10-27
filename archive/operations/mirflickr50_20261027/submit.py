"""Submit one data audit, two GPU preflights and five-epoch chains to epoch 50."""

import argparse
import json
from pathlib import Path
import subprocess


SLURM = Path("/usr/local/slurm/current/bin")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--submit", action="store_true", help="Otherwise print the commands only"
    )
    args = parser.parse_args()
    study = args.study.resolve()
    source = study / "source"
    script = source / "archive/operations/mirflickr50_20261027/run.sbatch"
    receipt = study / "jobs.json"
    if receipt.exists():
        raise FileExistsError(
            "Submission receipt exists; inspect it before adding jobs"
        )
    jobs = {}
    if args.submit:
        (study / "logs").mkdir(exist_ok=True)
        (study / "preflight").mkdir(exist_ok=True)

    def submit(key, action, *, gpu, dependency=None, method="", target=""):
        command = [
            str(SLURM / "sbatch"),
            "--parsable",
            "--account=monakhova",
            "--qos=normal",
            "--job-name=mir50-" + key,
            "--chdir=" + str(source),
            "--output=" + str(study / "logs" / (key + "_%j.out")),
            "--error=" + str(study / "logs" / (key + "_%j.err")),
            "--cpus-per-task=4",
            "--mem=" + ("48G" if gpu else "16G"),
            "--time=" + ("24:00:00" if action == "train" else "02:00:00"),
            "--partition=" + ("gpu" if gpu else "default_partition"),
        ]
        if gpu:
            command += [
                "--gres=gpu:1",
                "--constraint=(h100|h200|6000ada|a6000)&avx2",
                "--prefer=h100|h200|6000ada",
                "--requeue",
                "--exclude=monakhova-compute-01,nikola-compute-15,bhattacharjee-compute-02",
            ]
        else:
            command += ["--constraint=avx2"]
        if dependency:
            command += [
                "--dependency=afterok:" + dependency,
                "--kill-on-invalid-dep=yes",
            ]
        command += [
            str(script),
            str(source),
            str(study),
            str(args.manifest),
            action,
            method,
            str(target),
        ]
        if args.submit:
            job_id = subprocess.check_output(command, text=True).strip().split(";")[0]
            if not job_id.isdigit():
                raise ValueError("Unexpected sbatch output: " + job_id)
        else:
            job_id = "DRY_" + key
        jobs[key] = dict(job_id=job_id, command=command, dependency=dependency)
        if args.submit:
            temporary = receipt.with_suffix(".tmp")
            temporary.write_text(json.dumps(jobs, indent=2) + "\n")
            temporary.replace(receipt)
        print(json.dumps(dict(key=key, **jobs[key])), flush=True)
        return job_id

    audit = submit("audit", "audit", gpu=False)
    for method in ("im2im", "qutcc"):
        previous = submit(
            method + "-preflight", "model", gpu=True, dependency=audit, method=method
        )
        for epoch in range(5, 51, 5):
            previous = submit(
                f"{method}-{epoch:03d}",
                "train",
                gpu=True,
                dependency=previous,
                method=method,
                target=epoch,
            )


if __name__ == "__main__":
    main()
