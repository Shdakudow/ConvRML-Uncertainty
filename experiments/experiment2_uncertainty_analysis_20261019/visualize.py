"""Standalone scientific figures for the saved-example correlation analysis."""

import argparse, json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from scipy.ndimage import gaussian_filter
from analyze_saved_examples import METHODS, LABELS, source
from metrics import maps

COLORS = ["#277da8", "#f6a52d", "#7c61a8", "#2b9b83"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, required=True)
    a = p.parse_args()
    s = json.loads((a.results / "summary.json").read_text())
    out = a.results / "figures"
    out.mkdir(exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.facecolor": "white",
        }
    )
    with PdfPages(a.results / "Experiment2_LPIPS_Uncertainty_Error.pdf") as pdf:
        fig, axes = plt.subplots(2, 3, figsize=(14, 8), layout="constrained")
        for row, (camera, models) in enumerate(s["cameras"].items()):
            for col, (key, label) in enumerate(
                [
                    ("lpips", "LPIPS (AlexNet) ↓"),
                    ("pearson", "Pixel Pearson r ↑"),
                    ("spearman", "Pixel Spearman ρ ↑"),
                ]
            ):
                ax = axes[row, col]
                for i, method in enumerate(METHODS):
                    v = models[method]
                    if v["status"] != "complete":
                        ax.text(
                            i,
                            0.03,
                            "Pending",
                            ha="center",
                            rotation=90,
                            transform=ax.get_xaxis_transform(),
                        )
                        continue
                    stat = v["lpips"] if key == "lpips" else v["scales"]["0"][key]
                    mean = stat["mean"]
                    ci = stat["ci95"]
                    ax.bar(i, mean, color=COLORS[i], width=0.65)
                    ax.errorbar(
                        i,
                        mean,
                        yerr=[[mean - ci[0]], [ci[1] - mean]],
                        color="black",
                        capsize=3,
                        fmt="none",
                    )
                ax.set_xticks(
                    range(4),
                    ["Im2im", "Ensemble 5", "MC dropout", "QUTCC"],
                    rotation=20,
                )
                ax.set_ylabel(label)
                ax.set_title("RML" if camera == "rml" else "Normalized DiffuserCam")
                ax.grid(axis="y", alpha=0.2)
                ax.set_axisbelow(True)
                if key != "lpips":
                    ax.set_ylim(0, 1)
        fig.suptitle(
            "Experiment 2 · epoch 50 · first 50 test images\nUnsmoothed correlations; 95% image-bootstrap intervals. RML QUTCC pending calibration/export.",
            fontsize=14,
        )
        pdf.savefig(fig)
        fig.savefig(out / "summary.png", dpi=180)
        plt.close(fig)
        for camera, models in s["cameras"].items():
            loaded = {
                m: np.load(source(camera, m) / "examples.npz")
                for m in METHODS
                if models[m]["status"] == "complete"
            }
            rows = {
                m: json.loads((a.results / f"{camera}_{m}.json").read_text())["rows"]
                for m in loaded
            }
            for index in [0, 1, 5]:
                baseline = loaded["im2im"]
                truth = baseline["target"][index]
                measurement = baseline["measurement"][index]
                valid = {
                    m: maps(
                        np.clip(z["prediction"][index], 0, 1),
                        z["target"][index],
                        np.clip(z["lower"][index], 0, 1),
                        np.clip(z["upper"][index], 0, 1),
                    )
                    for m, z in loaded.items()
                }
                vmax = np.quantile(
                    np.concatenate(
                        [x.ravel() for pair in valid.values() for x in pair]
                    ),
                    0.995,
                )
                fig = plt.figure(figsize=(15, 14), layout="constrained")
                gs = fig.add_gridspec(5, 4, height_ratios=[0.8, 1, 1, 1, 1])
                for col, title, img in [
                    (0, "Measurement", measurement),
                    (1, "Ground truth", truth),
                ]:
                    ax = fig.add_subplot(gs[0, col])
                    ax.imshow(np.clip(img.transpose(1, 2, 0), 0, 1))
                    ax.set_title(title)
                    ax.axis("off")
                ax = fig.add_subplot(gs[0, 2:])
                ax.axis("off")
                ax.text(
                    0,
                    0.8,
                    "Error = mean RGB |prediction − truth|\nUncertainty = mean RGB (upper − lower)\nShared map scale across models and both quantities.\nDisplay saturates at the 99.5th percentile;\ncorrelations use original values, without smoothing.",
                    va="top",
                    fontsize=11,
                )
                for row, method in enumerate(METHODS, 1):
                    if method not in loaded:
                        ax = fig.add_subplot(gs[row, :])
                        ax.axis("off")
                        ax.text(
                            0.5,
                            0.5,
                            "QUTCC epoch-50 result pending calibration/export",
                            ha="center",
                        )
                        continue
                    z = loaded[method]
                    u, e = valid[method]
                    r = rows[method][index]
                    axes = [fig.add_subplot(gs[row, c]) for c in range(4)]
                    axes[0].imshow(
                        np.clip(z["prediction"][index].transpose(1, 2, 0), 0, 1)
                    )
                    axes[0].set_title(f"{LABELS[method]}\nLPIPS {r['lpips']:.5f}")
                    axes[0].axis("off")
                    for ax, value, title in [
                        (axes[1], e, "Absolute error"),
                        (axes[2], u, "Uncertainty width"),
                    ]:
                        im = ax.imshow(value, cmap="inferno", vmin=0, vmax=vmax)
                        ax.set_title(title)
                        ax.axis("off")
                        fig.colorbar(im, ax=ax, fraction=0.04)
                    rng = np.random.default_rng(20261019)
                    ix = rng.choice(u.size, 4096, replace=False)
                    axes[3].hexbin(
                        u.ravel()[ix],
                        e.ravel()[ix],
                        gridsize=35,
                        mincnt=1,
                        bins="log",
                        cmap="viridis",
                    )
                    axes[3].set_xlabel("Width")
                    axes[3].set_ylabel("Absolute error")
                    v = r["scales"]["0"]
                    axes[3].set_title(
                        f"Pearson {v['pearson']:.5f}\nSpearman {v['spearman']:.5f}"
                    )
                id = rows["im2im"][index]["id"]
                fig.suptitle(
                    f"{'RML' if camera=='rml' else 'Normalized DiffuserCam'} · test example {index+1} · ID {id}\nAll correlations use the full 256×256 map; scatter displays a fixed 4,096-pixel subset.",
                    fontsize=13,
                )
                pdf.savefig(fig)
                fig.savefig(out / f"{camera}_example_{index+1:02d}.png", dpi=140)
                plt.close(fig)
            fig, axes = plt.subplots(4, 4, figsize=(13, 13), layout="constrained")
            pairs = {
                m: maps(
                    np.clip(z["prediction"][0], 0, 1),
                    z["target"][0],
                    np.clip(z["lower"][0], 0, 1),
                    np.clip(z["upper"][0], 0, 1),
                )
                for m, z in loaded.items()
            }
            limit = np.quantile(
                np.concatenate([x.ravel() for pair in pairs.values() for x in pair]),
                0.995,
            )
            for i, m in enumerate(METHODS):
                if m not in pairs:
                    for ax in axes[i]:
                        ax.axis("off")
                        ax.text(0.5, 0.5, "Pending", ha="center")
                    continue
                u, e = pairs[m]
                for ax, image, title in zip(
                    axes[i],
                    [e, u, gaussian_filter(e, 1), gaussian_filter(u, 1)],
                    ["Error", "Width", "Error, Gaussian σ=1", "Width, Gaussian σ=1"],
                ):
                    im = ax.imshow(image, vmin=0, vmax=limit, cmap="inferno")
                    ax.set_title(title)
                    ax.set_xticks([])
                    ax.set_yticks([])
                axes[i, 0].set_ylabel(LABELS[m], fontsize=9)
            fig.suptitle(
                camera + " · example 1 · raw versus smoothed maps (same display scale)"
            )
            pdf.savefig(fig)
            fig.savefig(out / f"{camera}_raw_vs_smoothed.png", dpi=150)
            plt.close(fig)
            fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
            for ax, key in zip(axes, ["pearson", "spearman"]):
                for i, method in enumerate(METHODS):
                    if models[method]["status"] != "complete":
                        continue
                    scales = models[method]["scales"]
                    ax.plot(
                        [0, 0.5, 1, 2, 4],
                        [scales[str(x)][key]["mean"] for x in [0, 0.5, 1, 2, 4]],
                        "o-",
                        label=LABELS[method],
                        color=COLORS[i],
                    )
                ax.set_xlabel("Gaussian σ (pixels); σ=0 is the primary result")
                ax.set_ylabel("Mean image-wise " + key)
                ax.grid(alpha=0.2)
                ax.legend(fontsize=8)
            fig.suptitle(
                camera
                + " · secondary smoothing sensitivity (not used to select a winning scale)"
            )
            pdf.savefig(fig)
            fig.savefig(out / f"{camera}_smoothing.png", dpi=180)
            plt.close(fig)
    print("Wrote report and figures to", a.results)


if __name__ == "__main__":
    main()
