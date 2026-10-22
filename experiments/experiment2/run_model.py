"""Common entry point for the current four-model comparison.

QUTCC delegates to the included original-style source; the other methods retain
their corrected baseline trainer and asymmetric-RCPS milestone evaluator.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["smoke", "train", "evaluate"])
    parser.add_argument(
        "--method", choices=["qutcc", "im2im", "ensemble", "mc_dropout"], required=True
    )
    parser.add_argument("--camera", choices=["rml", "diffuser"], required=True)
    parser.add_argument(
        "--epoch",
        type=int,
        default=50,
        help="Absolute target/checkpoint epoch, not additional epochs",
    )
    parser.add_argument(
        "--member",
        type=int,
        choices=range(10),
        help="Ensemble member index for smoke/train",
    )
    parser.add_argument(
        "--members",
        type=int,
        choices=[5, 10],
        default=5,
        help="Number of fixed members for ensemble evaluation",
    )
    parser.add_argument("--dropout", type=float, choices=[0.05, 0.1, 0.2], default=0.05)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(os.environ.get("CONVRML_PROJECT", str(Path.home() / "ConvRML"))),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the command without importing models or running a job",
    )
    args = parser.parse_args(argv)
    if not 1 <= args.epoch <= 50:
        parser.error("--epoch must be in 1..50")
    if args.action == "evaluate" and args.epoch % 5:
        parser.error("Evaluation requires a saved five-epoch milestone")
    if args.method == "ensemble" and args.action != "evaluate" and args.member is None:
        parser.error(
            "Ensemble smoke/train requires --member 0..9; each member trains independently"
        )
    if args.member is not None and (
        args.method != "ensemble" or args.action == "evaluate"
    ):
        parser.error("--member is only for ensemble smoke/train")
    return args


def command(args):
    """Build a subprocess command without changing model or checkpoint protocols."""
    project = args.project_root.expanduser().resolve()
    env = dict(os.environ, CONVRML_PROJECT=str(project))
    if args.method == "qutcc":
        folder = ROOT / "qutcc"
        task = ["--task", str(["rml", "diffuser"].index(args.camera))]
        if args.action == "evaluate":
            cmd = [
                sys.executable,
                str(folder / "evaluate.py"),
                *task,
                "--epoch",
                str(args.epoch),
            ]
        else:
            cmd = [
                sys.executable,
                str(folder / "train.py"),
                *task,
                "--target-epoch",
                str(args.epoch),
            ]
            if args.action == "smoke":
                cmd.append("--smoke")
        return cmd, folder, env

    arm = args.method
    if args.method == "mc_dropout":
        arm = {0.05: "mc_005", 0.1: "mc_010", 0.2: "mc_020"}[args.dropout]
    elif args.method == "ensemble":
        arm = (
            f"ensemble_{args.member:02d}"
            if args.action != "evaluate"
            else f"ensemble{args.members}"
        )
    if args.action == "evaluate":
        cmd = [
            sys.executable,
            str(ROOT / "scripts/evaluate_interval_curves.py"),
            "--camera",
            args.camera,
            "--method",
            arm,
            "--epoch",
            str(args.epoch),
        ]
    else:
        cmd = [
            sys.executable,
            str(ROOT / "scripts/corrected_comparison_study.py"),
            args.action,
            "--camera",
            args.camera,
            "--arm",
            arm,
            "--epochs",
            str(args.epoch),
        ]
    return cmd, project, env


def main():
    args = arguments()
    cmd, cwd, env = command(args)
    print(
        json.dumps(
            dict(
                command=cmd,
                working_directory=str(cwd),
                project_root=env["CONVRML_PROJECT"],
                dry_run=args.dry_run,
            ),
            indent=2,
        ),
        flush=True,
    )
    if not args.dry_run:
        if not cwd.is_dir():
            raise SystemExit(f"Project directory does not exist: {cwd}")
        subprocess.run(cmd, cwd=cwd, env=env, check=True)


if __name__ == "__main__":
    main()
