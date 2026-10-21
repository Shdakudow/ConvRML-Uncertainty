"""Two independent resumed chains, five epochs/job, calibration at each milestone."""

import json, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SB = "/usr/local/slurm/current/bin/sbatch"
assert not (ROOT / "jobs.json").exists(), "Refusing duplicate submission"
subprocess.run(
    ["/home/jy2228/.conda/envs/convrml/bin/python", "test_continuation.py"],
    cwd=ROOT,
    check=True,
)
(ROOT / "logs").mkdir(exist_ok=True)
jobs = {}


def submit(name, args, deps=(), smoke=False):
    cmd = [SB, "--parsable", "--qos=normal", "--nice=0", "--job-name=" + name]
    if smoke:
        cmd += ["--time=00:30:00"]
    if deps:
        cmd += ["--dependency=afterok:" + ":".join(deps)]
    job = (
        subprocess.check_output(cmd + ["run.sbatch", *args], cwd=ROOT, text=True)
        .strip()
        .split(";")[0]
    )
    jobs[name] = job
    tmp = ROOT / "jobs.tmp"
    tmp.write_text(json.dumps(jobs, indent=2))
    tmp.replace(ROOT / "jobs.json")
    print(name, job, flush=True)
    return job


for task, method in enumerate(["qutcc", "im2im"]):
    previous = submit(
        "raw80-smoke-" + method,
        ["train.py", "--task", str(task), "--smoke"],
        smoke=True,
    )
    evaluation = None
    for epoch in range(55, 81, 5):
        previous = submit(
            f"raw80-{method}-{epoch}",
            ["train.py", "--task", str(task), "--target-epoch", str(epoch)],
            [previous],
        )
        evaluation = submit(
            f"raw80-eval-{method}-{epoch}",
            ["evaluate.py", "--task", str(task), "--epoch", str(epoch)],
            [previous] + ([evaluation] if evaluation else []),
        )
