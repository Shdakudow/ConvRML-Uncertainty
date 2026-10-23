"""Run one frozen-checkpoint milestone and distinguish calibration failures.

The task list is prepared separately; no missing checkpoint is substituted and
no training is performed. Known calibration infeasibility is a scientific
result. Interrupted jobs and other exceptions remain retryable job failures.
"""

import argparse
from collections import deque
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tasks", type=Path)
    parser.add_argument("index", type=int)
    args = parser.parse_args()
    task = json.loads(args.tasks.read_text())[args.index]
    assert task["index"] == args.index
    output = Path(task["output"])
    output.mkdir(parents=True, exist_ok=True)
    if (output / "report.json").exists():
        print("Already complete", output, flush=True)
        return
    failure = output / "failure.json"
    if (
        failure.exists()
        and json.loads(failure.read_text()).get("status") == "calibration_infeasible"
    ):
        print("Calibration infeasibility already recorded", output, flush=True)
        return
    print(json.dumps(task), flush=True)
    process = subprocess.Popen(
        task["command"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    tail = deque(maxlen=150)
    for line in process.stdout:
        print(line, end="", flush=True)
        tail.append(line)
    code = process.wait()
    if code == 0 and (output / "report.json").exists():
        failure.unlink(missing_ok=True)
        return
    detail = "".join(tail)
    known = [
        "No feasible extreme quantile endpoints",
        "Observed nonmonotone q response",
        "Quantile endpoints reversed",
        "Final calibration recheck failed",
    ]
    infeasible = next((message for message in known if message in detail), None)
    result = dict(
        status="calibration_infeasible" if infeasible else "execution_failed",
        group=task["group"],
        method=task["method"],
        epoch=task["epoch"],
        error=infeasible or f"Evaluator exited with code {code}",
        log_tail=detail,
        test_used_for_calibration=False,
    )
    temporary = failure.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(failure)
    if not infeasible:
        raise SystemExit(code or 1)
    print("Recorded calibration failure; no interval metrics invented", flush=True)


if __name__ == "__main__":
    main()
