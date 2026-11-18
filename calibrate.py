"""Calibrate a frozen QUTCC or Im2im checkpoint and evaluate held-out tests."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from calibration.calibration import KRCPS, QUTCCCalibration
from dataloader import ImagePairs
from models import make_model
from train import save_checkpoint


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@torch.inference_mode()
def evaluate(calibrator, loader, perceptual, device):
    """Measure RGB pixel coverage and image-averaged reconstruction metrics."""
    calibrator.model.eval()
    psnr = PeakSignalNoiseRatio(data_range=1.0, dim=(1, 2, 3)).to(device)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    covered = pixels = crossed = images = 0
    width = width_in_range = lpips = 0.0
    for measurement, target in loader:
        measurement, target = measurement.to(device), target.to(device)
        center, lower, upper = calibrator.predict(measurement)
        psnr.update(center, target)
        ssim.update(center, target)
        lpips += float(perceptual(center.clip(0, 1), target).sum())
        perceptual.reset()
        covered += int(((target >= lower) & (target <= upper)).sum())
        crossed += int((lower > upper).sum())
        width += float(F.relu(upper - lower).sum())
        # Length of the interval's intersection with the valid image range.
        width_in_range += float(F.relu(upper.clip(max=1) - lower.clip(min=0)).sum())
        pixels += target.numel()
        images += len(target)
    return dict(
        n_images=images, coverage=covered / pixels, crossing=crossed / pixels,
        mean_interval_width=width / pixels,
        mean_interval_width_in_image_range=width_in_range / pixels,
        psnr=float(psnr.compute()), ssim=float(ssim.compute()), lpips=lpips / images,
    )


def compact_parameters(parameters):
    """Keep the full tensor and split indices in parameters.pt, not the summary."""
    compact = {}
    for name, value in parameters.items():
        if isinstance(value, torch.Tensor):
            compact[name] = dict(shape=list(value.shape), minimum=float(value.min()),
                                 mean=float(value.mean()), maximum=float(value.max()))
        elif name.endswith("_indices"):
            compact[name + "_count"] = len(value)
        else:
            compact[name] = value
    return compact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(777)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = make_model(saved["method"], saved["height"], saved["width"], saved["dataset_size"])
    model.load_state_dict(saved["model"])
    model = model.to(device).eval()
    calibrator = QUTCCCalibration(model) if saved["method"] == "qutcc" else KRCPS(model)
    args.output.mkdir(parents=True, exist_ok=True)
    provenance = dict(
        checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=sha256(args.checkpoint),
        manifest=str(args.manifest.resolve()), manifest_sha256=sha256(args.manifest),
        method=saved["method"], selected_epoch=saved["epoch"],
        selected_through_epoch=saved.get("selected_through_epoch", saved["epoch"]),
        validation_ssim=saved.get("validation_ssim"), seed=777, batch_size=10,
        training_weights_changed=False, test_used_for_fitting=False,
    )
    parameters_file = args.output / "parameters.pt"
    fitted = (torch.load(parameters_file, map_location="cpu", weights_only=False)
              if parameters_file.exists() else None)
    if (fitted is not None
            and fitted["provenance"]["checkpoint_sha256"] == provenance["checkpoint_sha256"]
            and fitted["provenance"]["manifest_sha256"] == provenance["manifest_sha256"]):
        calibrator.parameters = fitted["parameters"]
    else:
        loader = DataLoader(ImagePairs(args.manifest, "calibration"),
                            batch_size=10, shuffle=False, num_workers=2)
        print(json.dumps(dict(phase="fitting", **provenance)), flush=True)
        calibrator.fit(loader, device)
        provenance["training_weights_changed"] = any(
            not torch.equal(value.detach().cpu(), saved["model"][name])
            for name, value in model.state_dict().items()
        )
        save_checkpoint(dict(parameters=calibrator.parameters, provenance=provenance), parameters_file)
    print(json.dumps(dict(phase="fitted", parameters=compact_parameters(calibrator.parameters))), flush=True)
    perceptual = LearnedPerceptualImagePatchSimilarity(
        net_type="alex", normalize=True, reduction="none"
    ).to(device).eval().requires_grad_(False)
    test_loader = DataLoader(ImagePairs(args.manifest, "test"),
                             batch_size=10, shuffle=False, num_workers=2)
    metrics = evaluate(calibrator, test_loader, perceptual, device)
    summary = dict(**provenance, calibration=compact_parameters(calibrator.parameters), test=metrics,
                   reconstruction_metrics="Raw PSNR/SSIM; LPIPS uses a clipped [0,1] copy",
                   coverage_definition="Fraction of RGB pixel-channel targets inside the interval",
                   width_definition="Nonnegative interval length; also reported after intersection with [0,1]")
    temporary = args.output / "summary.tmp"
    temporary.write_text(json.dumps(summary, indent=2) + "\n")
    temporary.replace(args.output / "summary.json")
    print(json.dumps(dict(phase="complete", **summary)), flush=True)


if __name__ == "__main__":
    main()
