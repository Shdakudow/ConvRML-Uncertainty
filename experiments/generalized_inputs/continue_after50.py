"""Start phase two only after a successful, complete 50-epoch predecessor."""

import argparse
import json
import os
from pathlib import Path
import shutil
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--method", choices=("qutcc", "im2im"), required=True)
    args = parser.parse_args()
    output = args.run_root / "results" / args.method
    archive = output / "COMPLETE_50epochs.json"
    complete = archive if archive.exists() else output / "COMPLETE.json"
    record = json.loads(complete.read_text())
    if (
        not record.get("passed")
        or record.get("epochs") != 50
        or record.get("method") != args.method
    ):
        raise ValueError("The matching 50-epoch predecessor has not completed.")
    predecessor = output / "config_before_100epochs.json"
    if not predecessor.exists():
        config = json.loads((output / "config.json").read_text())
        if config["epochs"] != 50 or config["pretrained"]:
            raise ValueError("Expected the existing 50-epoch scratch-trained model.")
        predecessor.write_text(json.dumps(config, indent=2) + "\n")
    if not (output / "latest_before_100epochs.pt").exists():
        os.link(output / "latest.pt", output / "latest_before_100epochs.pt")
    if not archive.exists():
        shutil.copy2(complete, archive)
    config = json.loads(predecessor.read_text())
    command = [
        sys.executable,
        "-u",
        str(Path(__file__).with_name("train.py")),
        "--method",
        args.method,
        "--manifest",
        config["manifest_path"],
        "--output",
        str(output),
        "--extension",
        str(predecessor),
        "--initial-config",
        str(output / "config_before_50epochs.json"),
        "--after50",
    ]
    # COMPLETE_50epochs.json remains the permanent phase-one completion record.
    current = output / "COMPLETE.json"
    if current.exists() and json.loads(current.read_text()).get("epochs") == 50:
        current.unlink()
    # Keep the trainer at the Slurm wrapper's child PID so checkpoint and cancel
    # signals reach its handlers directly, including after allocation requeues.
    os.execv(sys.executable, command)


if __name__ == "__main__":
    main()
