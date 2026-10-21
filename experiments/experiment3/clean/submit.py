"""Smoke-gated fresh chains, 2 epochs/job; independent calibration every 5 epochs."""

import json, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SB = "/usr/local/slurm/current/bin/sbatch"
if (ROOT / "jobs.json").exists():
    raise SystemExit("Existing job record: refusing duplicate submission")
(ROOT / "logs").mkdir(exist_ok=True)
subprocess.run(
    ["/home/jy2228/.conda/envs/convrml/bin/python", "test_protocol.py"],
    cwd=ROOT,
    check=True,
)
from data import audit

for camera in ["rml", "diffuser", "raw"]:
    print(camera, audit(camera))
jobs = {}


def submit(key, args, dependency=None):
    command = [SB, "--parsable", "--job-name=" + key]
    if key.startswith("paper-smoke-"):
        command += ["--time=00:20:00"]
    if dependency:
        command += ["--dependency=afterok:" + ":".join(dependency)]
    command += ["run.sbatch"] + args
    job = subprocess.check_output(command, cwd=ROOT, text=True).strip().split(";")[0]
    jobs[key] = job
    tmp = ROOT / "jobs.tmp"
    tmp.write_text(json.dumps(jobs, indent=2))
    tmp.replace(ROOT / "jobs.json")
    print(key, job, flush=True)
    return job


for task, label in enumerate(["rml-q", "diff-q", "raw-q", "raw-im2im"]):
    smoke = submit("paper-smoke-" + label, ["train.py", "--task", str(task), "--smoke"])
    previous = smoke
    milestones = {}
    for epoch in range(2, 51, 2):
        previous = submit(
            f"paper-{label}-{epoch:02d}",
            ["train.py", "--task", str(task), "--target-epoch", str(epoch)],
            [previous],
        )
        milestones[epoch] = previous
    evaluation = None
    for epoch in range(5, 51, 5):
        deps = [milestones[epoch + (epoch % 2)]]
        if evaluation:
            deps.append(evaluation)
        evaluation = submit(
            f"paper-eval-{label}-{epoch:02d}",
            ["evaluate.py", "--task", str(task), "--epoch", str(epoch)],
            deps,
        )
