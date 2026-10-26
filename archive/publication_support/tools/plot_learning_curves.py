"""Plot evaluated five-epoch reconstruction/interval curves without GPU inference.

Accept latest QUTCC/raw report directories and corrected baseline curve reports.
Raw and clipped metrics are never silently substituted for one another.
"""

import argparse
import csv
import json
import math
from pathlib import Path

METRICS = ("ssim", "psnr", "width", "coverage")


def read_series(label, directory, metric_space="clipped"):
    """Read completed reports and reject duplicate or ambiguous milestones."""
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError(f"Report directory does not exist: {directory}")
    files = sorted(
        set(directory.rglob("report.json")) | set(directory.glob("epoch_*.json"))
    )
    rows = []
    seen = set()
    for path in files:
        report = json.loads(path.read_text())
        if "metrics" not in report or "epoch" not in report:
            continue
        epoch = report["epoch"]
        if not isinstance(epoch, int) or epoch < 5 or epoch % 5:
            continue
        if epoch in seen:
            raise ValueError(
                f"{label}: duplicate epoch {epoch}; choose a single model/camera directory"
            )
        metrics = report["metrics"]
        if metric_space in metrics:
            metrics = metrics[metric_space]
        elif "raw" in metrics or "clipped" in metrics:
            raise ValueError(f"{path}: no {metric_space} metrics")
        elif report.get("clipped") is not (metric_space == "clipped"):
            raise ValueError(
                f"{path}: cannot establish {metric_space} metric convention"
            )
        for key in METRICS:
            if key not in metrics or not math.isfinite(float(metrics[key])):
                raise ValueError(f"{path}: missing/nonfinite {key}")
        calibration = report.get("calibration") or {}
        factors = report.get("factors") or {}
        scales = calibration.get("scales") or factors.get("rcps_hb") or factors
        count = (
            report.get("test_count")
            or report.get("full_test_count")
            or report.get("count")
        )
        rows.append(
            dict(
                series=label,
                camera=report.get("camera", ""),
                method=report.get("method", label),
                epoch=epoch,
                metric_space=metric_space,
                test_count=count,
                calibration_count=report.get(
                    "calibration_count", calibration.get("n_images", "")
                ),
                **{key: float(metrics[key]) for key in METRICS},
                lower_q=calibration.get("lower_q", ""),
                upper_q=calibration.get("upper_q", ""),
                lower_scale=scales.get("lower", ""),
                upper_scale=scales.get("upper", ""),
                source=str(path.resolve()),
            )
        )
        seen.add(epoch)
    if not rows:
        raise ValueError(f"No completed five-epoch metric reports found in {directory}")
    return sorted(rows, key=lambda row: row["epoch"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--series", action="append", required=True, metavar="LABEL=DIRECTORY"
    )
    parser.add_argument("--metric-space", choices=["raw", "clipped"], default="clipped")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    labels = []
    for specification in args.series:
        if "=" not in specification:
            parser.error("--series must be LABEL=DIRECTORY")
        label, directory = specification.split("=", 1)
        if not label or label in labels:
            parser.error("Each series requires a distinct nonempty label")
        labels.append(label)
        rows.extend(read_series(label, directory, args.metric_space))
    cameras = {row["camera"] for row in rows if row["camera"]}
    if len(cameras) > 1:
        parser.error("Plot cameras separately; the supplied reports mix datasets")
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "learning_curves.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = list(range(5, max(row["epoch"] for row in rows) + 1, 5))
    figure, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for axis, key, title in zip(
        axes.flat,
        METRICS,
        ["SSIM", "PSNR (dB)", "Mean interval width", "Actual test coverage (%)"],
    ):
        for label in labels:
            by_epoch = {row["epoch"]: row for row in rows if row["series"] == label}
            values = [
                by_epoch[epoch][key] if epoch in by_epoch else math.nan
                for epoch in epochs
            ]
            if key == "coverage":
                values = [value * 100 for value in values]
            axis.plot(epochs, values, marker="o", label=label)
        axis.set(xlabel="Training / adaptation epoch", ylabel=title, xticks=epochs)
        axis.grid(alpha=0.25)
        axis.legend()
    camera = next(iter(cameras), "Dataset supplied by caller")
    figure.suptitle(
        f"{camera} · {args.metric_space} metrics · evaluated milestones only"
    )
    figure.savefig(args.output / "learning_curves.png", dpi=180)
    plt.close(figure)
    print(f"Saved {len(rows)} evaluated points to {args.output}")


if __name__ == "__main__":
    main()
