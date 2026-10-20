"""Restore failed/cancelled corrected-study jobs while retaining completed work and six lanes."""

import argparse
import collections
import json
import subprocess
from pathlib import Path

ROOT = Path("results/corrected_comparison_20261006")
SLURM = "/usr/local/slurm/current/bin/"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    record = ROOT / "jobs.json"
    config = json.loads(record.read_text())
    ids = ",".join(j["job"] for j in config["jobs"])
    wanted = set(ids.split(","))
    raw = subprocess.check_output(
        [SLURM + "sacct", "-X", "-n", "-P", "-j", ids, "--format=JobID,State"],
        text=True,
    )
    states = {
        line.split("|")[0]: line.split("|")[1].split()[0]
        for line in raw.splitlines()
        if "|" in line and line.split("|")[0] in wanted
    }
    print(json.dumps(dict(states=collections.Counter(states.values()))), flush=True)
    if not args.apply:
        return
    backup = ROOT / "jobs_before_runtime_recovery_20261009.json"
    if not backup.exists():
        backup.write_text(record.read_text())
    mapping = {}
    audit = []
    for job in config["jobs"]:
        old = job["job"]
        state = states.get(old)
        if state in ("COMPLETED", "RUNNING"):
            continue
        if state not in (
            "FAILED",
            "CANCELLED",
            "PENDING",
            "TIMEOUT",
            "NODE_FAIL",
            "OUT_OF_MEMORY",
        ):
            raise RuntimeError(f"Unrecognized job state {old}: {state}")
        original_deps = (job.get("dependency") or "").split(":")
        dependencies = [
            mapping.get(d, d)
            for d in original_deps
            if d and states.get(d) != "COMPLETED"
        ]
        dependency = ":".join(dependencies)
        if state == "PENDING":
            if any(d in mapping for d in original_deps):
                subprocess.run(
                    [
                        SLURM + "scontrol",
                        "update",
                        "JobId=" + old,
                        "Dependency=afterok:" + dependency,
                    ],
                    check=True,
                )
                job["dependency"] = dependency
            continue
        stage = job["stage"]
        command = [SLURM + "sbatch", "--parsable"]
        if dependency:
            command += [
                "--dependency=afterok:" + dependency,
                "--kill-on-invalid-dep=yes",
            ]
        if stage == "smoke":
            command += ["--time=00:20:00"]
        cli = []
        if stage in ("train", "smoke"):
            cli = ["--task", str(job["task"])]
            if job.get("epoch") is not None:
                cli += ["--epochs", str(job["epoch"])]
        elif stage == "tune":
            # Submitted in camera/rate order after the training barrier.
            cli = [
                "--task",
                str(
                    sum(
                        j["stage"] == "tune"
                        for j in config["jobs"][: config["jobs"].index(job)]
                    )
                ),
            ]
        elif stage == "select":
            index = sum(
                j["stage"] == "select"
                for j in config["jobs"][: config["jobs"].index(job)]
            )
            cli = ["--camera", ("rml", "diffuser")[index]]
        elif stage == "evaluate":
            index = sum(
                j["stage"] == "evaluate"
                for j in config["jobs"][: config["jobs"].index(job)]
            )
            cli = ["--task", str(index // 2 + (6 if index % 2 else 0))]
        else:
            raise RuntimeError(stage)
        new = (
            subprocess.check_output(
                command + ["scripts/corrected_comparison.sbatch", stage] + cli,
                text=True,
            )
            .strip()
            .split(";")[0]
        )
        mapping[old] = new
        audit.append(
            dict(
                old=old,
                new=new,
                stage=stage,
                task=job.get("task"),
                dependency=dependency,
            )
        )
        job.setdefault("previous_job_ids", []).append(old)
        job.update(job=new, dependency=dependency or None)
        temporary = record.with_suffix(".tmp")
        temporary.write_text(json.dumps(config, indent=2))
        temporary.replace(record)
        (ROOT / "runtime_recovery_20261009.json").write_text(
            json.dumps(audit, indent=2)
        )
        print(json.dumps(audit[-1]), flush=True)
    print(json.dumps(dict(replaced=len(audit))), flush=True)


if __name__ == "__main__":
    main()
