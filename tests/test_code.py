"""Behavior checks live here rather than as guards in the implementation."""

import gc
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ConvNeXt.backbone import ConvNeXt, ConvRecon
from Models.common import pinball_loss
from Models import make_model
from Models.qutcc import QUTCC
from Models.im2im import Im2im
from Models.ensemble import DeepEnsemble, DeepEnsembleMember
from Models.mcdropout import MCDropout
from dataloader import ImagePairs
from Calibration.calibration import KRCPS, QUTCCCalibration
from Calibration.krcps_optimization import SETTINGS
from train import learning_rate, train_epoch, validate


def small_backbone(channels=3, height=33, width=35):
    # Exercise the real encoder/decoder interfaces with smaller test-only widths.
    backbone = ConvRecon(3, height, width, out_channels=channels)
    backbone.encoder = ConvNeXt(depths=[1, 1, 1, 1], dims=[8, 16, 32, 64])
    backbone.decoder.FNN1007.f = nn.Sequential(
        nn.Conv2d(120, 8, 3, padding=1),
        nn.BatchNorm2d(8),
        nn.ReLU(),
        nn.Conv2d(8, channels, 1),
    )
    return backbone


class ReconstructionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(10)

    def tearDown(self):
        gc.collect()

    def test_all_python_files_parse(self):
        root = Path(__file__).resolve().parents[1]
        for path in root.rglob("*.py"):
            if path.relative_to(root).parts[0] == "archive":
                continue
            compile(path.read_text(), str(path), "exec")

    def test_ten_member_initializations_differ_and_repeat_with_seed_777(self):
        # Small layers isolate initialization from full-size GPU requirements.
        def backbone(channels, height, width, out_channels):
            return nn.Conv2d(channels, out_channels, kernel_size=1)

        with patch("Models.ConvRecon", side_effect=backbone):
            torch.manual_seed(777)
            model = make_model("ensemble", 33, 35, 10)
            torch.manual_seed(777)
            repeated = make_model("ensemble", 33, 35, 10)
        self.assertEqual(len(model.members), 10)
        weights = [member.baseModel.weight for member in model.members]
        self.assertEqual(len({weight.data_ptr() for weight in weights}), 10)
        for index, weight in enumerate(weights):
            for other in weights[:index]:
                self.assertFalse(torch.equal(weight, other))
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, repeated.state_dict()[name])

    def test_loader_preserves_pair_values_and_size(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            Image.fromarray(np.full((33, 35, 3), 51, dtype=np.uint8)).save(
                root / "input.png"
            )
            Image.fromarray(np.full((33, 35, 3), 204, dtype=np.uint8)).save(
                root / "target.png"
            )
            row = {
                "id": "scene",
                "measurement": {"path": "input.png"},
                "target": {"path": str(root / "target.png")},
            }
            (root / "split.json").write_text(json.dumps({"roles": {"train": [row]}}))
            measurement, target = ImagePairs(root / "split.json", "train")[0]
            self.assertEqual(measurement.shape, (3, 33, 35))
            torch.testing.assert_close(measurement, torch.full_like(measurement, 0.2))
            torch.testing.assert_close(target, torch.full_like(target, 0.8))

    def test_loader_corrects_original_mirflickr_arrays(self):
        # Unequal rows and channels detect orientation and BGR/RGB mistakes.
        rgb = np.array(
            [[[0.0, 0.2, 0.4], [0.6, 0.8, 1.2]],
             [[-0.1, 0.1, 0.3], [0.5, 0.7, 0.9]]],
            dtype=np.float64,
        )
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            np.save(root / "measurement.npy", (rgb * 0.9)[::-1, :, ::-1])
            np.save(root / "target.npy", rgb[::-1, :, ::-1])
            row = dict(
                measurement=dict(path="measurement.npy"),
                target=dict(path=str(root / "target.npy")),
            )
            manifest = root / "split.json"
            manifest.write_text(json.dumps(dict(roles=dict(train=[row]))))
            measurement, target = ImagePairs(manifest, "train")[0]
            expected = torch.from_numpy(rgb.clip(0, 1)).permute(2, 0, 1).float()
            self.assertEqual(measurement.dtype, torch.float32)
            self.assertEqual(target.dtype, torch.float32)
            torch.testing.assert_close(measurement, expected)
            torch.testing.assert_close(target, expected)

    def test_loader_batches_full_resolution_mirflickr_arrays(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            pixels = np.random.default_rng(777).random((270, 480, 3)).astype(np.float32)
            np.save(root / "measurement.npy", (pixels * 0.9)[::-1, :, ::-1])
            np.save(root / "target.npy", pixels[::-1, :, ::-1])
            row = dict(
                measurement=dict(path="measurement.npy"),
                target=dict(path="target.npy"),
            )
            manifest = root / "split.json"
            manifest.write_text(json.dumps(dict(roles=dict(train=[row] * 10))))
            loader = DataLoader(ImagePairs(manifest, "train"), batch_size=10)
            measurement, target = next(iter(loader))
            self.assertEqual(measurement.shape, (10, 3, 270, 480))
            self.assertEqual(target.shape, measurement.shape)
            torch.testing.assert_close(measurement, target)

    def test_real_convnext_reconstruction(self):
        backbone = ConvRecon(3, 33, 35).eval()
        with torch.no_grad():
            self.assertEqual(backbone(torch.rand(1, 3, 33, 35)).shape, (1, 3, 33, 35))

    def test_qutcc_keeps_backbone_at_initialization_and_learns_q(self):
        backbone = small_backbone().eval()
        model = QUTCC(backbone).eval()
        image = torch.rand(2, 3, 33, 35)
        with torch.no_grad():
            torch.testing.assert_close(model.center(image), backbone(image))
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        model.train()
        loss = model.loss(image, torch.rand_like(image))
        loss.backward()
        self.assertGreater(
            float(model.encoder_film[0][0].projection[-1].weight.grad.abs().sum()), 0
        )
        optimizer.step()
        model.eval()
        with torch.no_grad():
            self.assertGreater(
                float((model(image, 0.1) - model(image, 0.9)).abs().sum()), 0
            )
        # Pinball broadcasts a scalar or one q per image.
        torch.testing.assert_close(
            pinball_loss(image, image + 1, 0.1), torch.tensor(0.1)
        )
        torch.testing.assert_close(
            pinball_loss(image, image + 1, torch.tensor([0.1, 0.9])), torch.tensor(0.5)
        )

    def test_im2im_heads_and_backward(self):
        model = Im2im(small_backbone(9))
        image = torch.rand(2, 3, 33, 35)
        output = model(image)
        self.assertEqual(output.shape, (2, 3, 3, 33, 35))
        center, lower, upper = model.bounds(image)
        torch.testing.assert_close(center, output[:, 1])
        torch.testing.assert_close(lower, output[:, 0])
        torch.testing.assert_close(upper, output[:, 2])
        model.loss(image, torch.rand_like(image)).backward()
        self.assertTrue(
            torch.isfinite(model.backbone.decoder.FNN1007.f[-1].weight.grad).all()
        )

    def test_ensemble_rgb_variance_and_total_variance(self):
        class Constant(nn.Module):
            def __init__(self, mean):
                super().__init__()
                self.mean = mean

            def forward(self, image):
                return torch.full_like(image, self.mean), torch.full_like(image, 2)

        image = torch.rand(2, 3, 33, 35)
        ensemble = DeepEnsemble([Constant(1), Constant(3)])
        mean, variance = ensemble.moments(image)
        torch.testing.assert_close(mean, torch.full_like(image, 2))
        torch.testing.assert_close(variance, torch.full_like(image, 3))
        member = DeepEnsembleMember(small_backbone(6))
        mean, variance = member(image)
        self.assertEqual(mean.shape, image.shape)
        self.assertTrue((variance > 0).all())
        member.loss(image, torch.rand_like(image)).backward()

    def test_dropout_samples_and_normalization_modes(self):
        model = MCDropout(small_backbone(), dataset_size=10).train()
        modes = {name: module.training for name, module in model.named_modules()}
        before = [
            layer.running_mean.clone()
            for layer in model.modules()
            if isinstance(layer, nn.BatchNorm2d)
        ]
        with torch.no_grad():
            samples = model.predict_samples(torch.rand(2, 3, 33, 35))
        self.assertEqual(samples.shape, (30, 2, 3, 33, 35))
        self.assertGreater(float(samples.var(0).mean()), 0)
        self.assertEqual(
            modes, {name: module.training for name, module in model.named_modules()}
        )
        after = [
            layer.running_mean
            for layer in model.modules()
            if isinstance(layer, nn.BatchNorm2d)
        ]
        for old, new in zip(before, after):
            torch.testing.assert_close(old, new)
        with patch.object(
            model.baseModel, "forward", wraps=model.baseModel.forward
        ) as forward:
            loss = model.loss(torch.rand(2, 3, 33, 35), torch.rand(2, 3, 33, 35))
        self.assertEqual(forward.call_count, 1)
        loss.backward()
        self.assertIsNotNone(next(model.parameters()).grad)

    def test_dropout_center_and_bounds_use_the_same_30_pass_mean(self):
        model = MCDropout(small_backbone(), dataset_size=10).eval()
        image = torch.rand(2, 3, 33, 35)
        with torch.inference_mode():
            torch.manual_seed(777)
            expected = model.predict_samples(image).mean(dim=0)
            torch.manual_seed(777)
            with patch.object(
                model.baseModel, "forward", wraps=model.baseModel.forward
            ) as forward:
                center = model.center(image)
            self.assertEqual(forward.call_count, 30)
            torch.manual_seed(777)
            interval_center, _, _ = model.bounds(image)
        torch.testing.assert_close(center, expected)
        torch.testing.assert_close(interval_center, expected)

    def test_calibrated_prediction_disables_gradients_for_both_methods(self):
        class Bounds(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.tensor(0.5))

            def bounds(self, image, *quantiles):
                center = image * self.weight
                return center, center - 0.1, center + 0.1

        for method in (KRCPS, QUTCCCalibration):
            with self.subTest(method=method.__name__):
                model = Bounds().eval()
                calibrator = method(model)
                calibrator.parameters = dict(
                    lambda_map=torch.zeros(4, 4), lower_q=0.05, upper_q=0.95
                )
                with torch.enable_grad():
                    outputs = calibrator.predict(torch.ones(2, 3, 4, 4))
                for output in outputs:
                    self.assertFalse(output.requires_grad)
                    self.assertIsNone(output.grad_fn)
                self.assertTrue(model.weight.requires_grad)
                model.bounds(torch.ones(2, 3, 4, 4))[0].sum().backward()
                self.assertIsNotNone(model.weight.grad)

    def test_schedule_endpoints_and_decay(self):
        self.assertAlmostEqual(learning_rate(0, 10001), 1e-5)
        self.assertAlmostEqual(learning_rate(1800, 10001), 1e-4)
        self.assertAlmostEqual(learning_rate(10000, 10001), 1e-9)
        self.assertGreater(learning_rate(6000, 10001), learning_rate(9000, 10001))

    def test_ensemble_member_steps_match_joint_mean_nll_update(self):
        import copy

        model = DeepEnsemble(
            [
                DeepEnsembleMember(small_backbone(6)),
                DeepEnsembleMember(small_backbone(6)),
            ]
        )
        reference = copy.deepcopy(model)
        image, target = torch.rand(2, 3, 33, 35), torch.rand(2, 3, 33, 35)
        reference_optimizer = torch.optim.AdamW(
            reference.parameters(), lr=learning_rate(0, 100), weight_decay=1e-4
        )
        reference.loss(image, target).backward()
        reference_optimizer.step()
        optimizer = torch.optim.AdamW(model.parameters())
        train_epoch(
            model,
            DataLoader(TensorDataset(image, target), batch_size=2),
            optimizer,
            "cpu",
            0,
            100,
        )
        for name, actual in model.state_dict().items():
            torch.testing.assert_close(actual, reference.state_dict()[name])
        self.assertTrue(all(state["step"] == 1 for state in optimizer.state.values()))

    def test_ensemble_shuffles_batch_order_independently_and_reproducibly(self):
        class RecordingMember(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.tensor(0.1))
                self.seen = []

            def loss(self, image, target):
                self.seen.append(
                    (tuple(image[:, 0].tolist()), tuple(target[:, 0].tolist()))
                )
                return (self.weight * image - target).square().mean()

        image = torch.arange(43, dtype=torch.float32).reshape(-1, 1)
        loader = DataLoader(TensorDataset(image, image + 100), batch_size=10)
        expected_batches = [
            tuple(range(start, min(start + 10, 43)))
            for start in range(0, 43, 10)
        ]

        def run_epoch():
            torch.manual_seed(777)
            model = DeepEnsemble([RecordingMember() for _ in range(3)])
            optimizer = torch.optim.AdamW(model.parameters())
            with patch("train.learning_rate", wraps=learning_rate) as schedule:
                loss, step = train_epoch(model, loader, optimizer, "cpu", 7, 100)
            self.assertTrue(np.isfinite(loss))
            self.assertEqual(step, 12)
            self.assertEqual(
                [call.args[0] for call in schedule.call_args_list],
                list(range(7, 12)) * 3,
            )
            self.assertTrue(all(state["step"] == 5 for state in optimizer.state.values()))
            orders = []
            for member in model.members:
                order = [batch for batch, _ in member.seen]
                self.assertCountEqual(order, expected_batches)
                for batch, targets in member.seen:
                    self.assertEqual(targets, tuple(value + 100 for value in batch))
                orders.append(order)
            self.assertGreater(len({tuple(order) for order in orders}), 1)
            return model, optimizer, orders

        model, optimizer, orders = run_epoch()
        _, _, repeated_orders = run_epoch()
        self.assertEqual(orders, repeated_orders)
        for member in model.members:
            member.seen.clear()
        _, step = train_epoch(model, loader, optimizer, "cpu", 12, 100)
        self.assertEqual(step, 17)
        next_orders = [[batch for batch, _ in member.seen] for member in model.members]
        self.assertNotEqual(orders, next_orders)
        self.assertTrue(all(state["step"] == 10 for state in optimizer.state.values()))

    def test_training_validation_and_checkpoint_roundtrip(self):
        class PerceptualStub(nn.Module):
            # Keep this loop test offline; actual PSNR/SSIM use torchmetrics.
            def forward(self, prediction, target):
                return (prediction - target).abs().mean((1, 2, 3))

            def reset(self):
                pass

        model = Im2im(small_backbone(9))
        dataset = TensorDataset(torch.rand(11, 3, 33, 35), torch.rand(11, 3, 33, 35))
        loader = DataLoader(dataset, batch_size=10)
        optimizer = torch.optim.AdamW(model.parameters())
        loss, step = train_epoch(model, loader, optimizer, "cpu", 0, 100)
        self.assertTrue(np.isfinite(loss))
        self.assertEqual(step, 2)
        scores = validate(model, loader, PerceptualStub(), "cpu")
        self.assertEqual(set(scores), {"psnr", "ssim", "lpips"})
        self.assertTrue(all(np.isfinite(value) for value in scores.values()))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.pt"
            torch.save(model.state_dict(), path)
            other = Im2im(small_backbone(9)).eval()
            other.load_state_dict(torch.load(path, weights_only=True))
            with torch.no_grad():
                torch.testing.assert_close(
                    model.center(dataset.tensors[0]), other.center(dataset.tensors[0])
                )

    def test_calibration_uses_frozen_model_and_separate_parameters(self):
        class Bounds(nn.Module):
            def __init__(self):
                super().__init__()
                self.offset = nn.Parameter(torch.tensor(0.5))

            def bounds(self, image):
                center = torch.zeros_like(image) + self.offset
                return center, center - 0.1, center + 0.1

        model = Bounds()
        target = torch.linspace(0.2, 0.8, 140 * 3 * 8 * 8).reshape(140, 3, 8, 8)
        loader = DataLoader(
            TensorDataset(torch.zeros_like(target), target), batch_size=10
        )
        calibrator = KRCPS(model)
        # Reduce only the test problem; production settings come from the
        # existing code-for-quant configuration.
        with patch.dict(SETTINGS, {"prob_size": 16, "gamma": [0.5]}):
            fit = calibrator.fit(loader, "cpu")
        self.assertIs(fit, calibrator.parameters)
        center, lower, upper = calibrator.predict(torch.zeros(2, 3, 8, 8))
        self.assertTrue((lower <= center).all() and (center <= upper).all())
        self.assertTrue(fit["certificate_passed"])
        self.assertGreaterEqual(fit["calibration_coverage"], 0.9)
        self.assertEqual(float(model.offset.detach()), 0.5)
        self.assertIsNone(model.offset.grad)
        self.assertEqual(fit["confidence"], 0.9)
        self.assertEqual(fit["alpha"], 0.1)
        self.assertEqual(fit["n_optimization_images"], 10)
        self.assertEqual(fit["n_images"], 130)
        self.assertFalse(
            set(fit["optimization_indices"]) & set(fit["calibration_indices"])
        )

    def test_hb_threshold_search_uses_40_iterations(self):
        calibrator = KRCPS(nn.Identity())
        with patch.object(
            calibrator, "_hb_pvalue", wraps=calibrator._hb_pvalue
        ) as bound:
            threshold = calibrator._accepted_risk(100)
        self.assertEqual(bound.call_count, 40)
        self.assertLess(calibrator._hb_pvalue(threshold, 100), 0.1)

    def test_validation_uses_raw_psnr_ssim_and_lpips_image_range(self):
        class Constant(nn.Module):
            def center(self, image):
                return torch.full_like(image, 1.5)

        class Perceptual(nn.Module):
            def forward(self, prediction, target):
                self.prediction = prediction
                return (prediction - target).abs().mean((1, 2, 3))

            def reset(self):
                pass

        image = torch.ones(2, 3, 33, 35)
        perceptual = Perceptual()
        scores = validate(
            Constant(),
            DataLoader(TensorDataset(image, image), batch_size=10),
            perceptual,
            "cpu",
        )
        self.assertAlmostEqual(scores["psnr"], 6.0206, places=4)
        self.assertLess(scores["ssim"], 1)
        self.assertEqual(float(perceptual.prediction.max()), 1)

    def test_validation_averages_images_and_does_not_mix_epochs(self):
        class Reconstruction(nn.Module):
            def center(self, image):
                return image

        class Perceptual(nn.Module):
            def forward(self, prediction, target):
                return (prediction - target).abs().mean((1, 2, 3))

            def reset(self):
                pass

        target = torch.ones(3, 3, 33, 35)
        prediction = target * torch.tensor([0.9, 0.5, 0.7]).reshape(3, 1, 1, 1)
        loader = DataLoader(TensorDataset(prediction, target), batch_size=2)
        result = validate(Reconstruction(), loader, Perceptual(), "cpu")
        expected = float(
            (-10 * (prediction - target).square().mean((1, 2, 3)).log10()).mean()
        )
        self.assertAlmostEqual(result["psnr"], expected, places=5)
        repeated = validate(Reconstruction(), loader, Perceptual(), "cpu")
        self.assertEqual(result, repeated)

    def test_qutcc_calibration_selects_quantile_inputs(self):
        class QuantileModel(QUTCC):
            def __init__(self):
                nn.Module.__init__(self)

            def forward(self, image, q=0.5):
                return torch.full_like(image, float(q))

        target = torch.linspace(0, 1, 1200).reshape(100, 3, 2, 2)
        loader = DataLoader(
            TensorDataset(torch.zeros_like(target), target), batch_size=10
        )
        calibrator = QUTCCCalibration(QuantileModel())
        fit = calibrator.fit(loader, "cpu")
        _, lower, upper = calibrator.predict(torch.zeros(2, 3, 2, 2))
        torch.testing.assert_close(lower, torch.full_like(lower, fit["lower_q"]))
        torch.testing.assert_close(upper, torch.full_like(upper, fit["upper_q"]))
        self.assertLess(fit["lower_q"], 0.05)
        self.assertGreater(fit["upper_q"], 0.95)
        self.assertTrue(fit["target_met"])

    def test_real_lpips_and_training_entrypoint(self):
        import train

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            image = np.random.default_rng(10).integers(
                0, 256, (64, 64, 3), dtype=np.uint8
            )
            Image.fromarray(image).save(root / "input.png")
            Image.fromarray(image[::-1]).save(root / "target.png")
            rows = [
                dict(
                    id=str(index),
                    measurement=dict(path="input.png"),
                    target=dict(path="target.png"),
                )
                for index in range(4)
            ]
            manifest = root / "split.json"
            manifest.write_text(
                json.dumps(dict(roles=dict(train=rows[:2], val=rows[2:])))
            )
            output = root / "results"

            def test_model(name, height, width, dataset_size):
                return Im2im(small_backbone(9, height, width))

            with patch(
                "sys.argv",
                [
                    "train.py",
                    "--manifest",
                    str(manifest),
                    "--model",
                    "im2im",
                    "--output",
                    str(output),
                ],
            ), patch.object(train, "EPOCHS", 10), patch.object(
                train, "make_model", test_model
            ):
                train.main()
            history = json.loads((output / "history.json").read_text())
            self.assertEqual(len(history), 10)
            self.assertTrue(np.isfinite(history[-1]["lpips"]))
            self.assertLess(history[-1]["weight_decay"], history[0]["weight_decay"])
            checkpoint = torch.load(
                output / "best_through_epoch_010.pt", weights_only=False
            )
            best = max(history, key=lambda row: row["ssim"])
            self.assertEqual(checkpoint["epoch"], best["epoch"])
            self.assertEqual(checkpoint["validation_ssim"], best["ssim"])
            self.assertEqual(checkpoint["selected_through_epoch"], 10)
            self.assertEqual(checkpoint["method"], "im2im")

    def test_best_checkpoints_keep_cumulative_winner_and_frozen_weights(self):
        import train

        model = nn.Linear(1, 1, bias=False)
        dataset = TensorDataset(torch.zeros(10, 3, 2, 2), torch.zeros(10, 3, 2, 2))
        loader = DataLoader(dataset, batch_size=10)
        scores = [0.1] * 30
        scores[2] = 0.8
        scores[25] = 0.95

        def train_next_epoch(
            model, loader, optimizer, device, global_step, total_steps
        ):
            with torch.no_grad():
                model.weight.fill_(global_step + 1)
            return 0.0, global_step + 1

        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            with patch(
                "sys.argv",
                [
                    "train.py",
                    "--manifest",
                    "unused.json",
                    "--model",
                    "im2im",
                    "--output",
                    folder,
                ],
            ), patch.object(train, "EPOCHS", 30), patch.object(
                train, "get_loaders", return_value=(loader, loader)
            ), patch.object(
                train, "make_model", return_value=model
            ), patch.object(
                train,
                "LearnedPerceptualImagePatchSimilarity",
                return_value=nn.Identity(),
            ), patch.object(
                train, "train_epoch", side_effect=train_next_epoch
            ), patch.object(
                train,
                "validate",
                side_effect=[dict(psnr=0.0, ssim=score, lpips=0.0) for score in scores],
            ), patch(
                "builtins.print"
            ):
                train.main()
            for milestone, winner in [(10, 3), (20, 3), (30, 26)]:
                saved = torch.load(
                    output / f"best_through_epoch_{milestone:03d}.pt",
                    weights_only=False,
                )
                self.assertEqual(saved["epoch"], winner)
                self.assertEqual(saved["selected_through_epoch"], milestone)
                self.assertEqual(saved["validation_ssim"], scores[winner - 1])
                torch.testing.assert_close(
                    saved["model"]["weight"], torch.full((1, 1), float(winner))
                )
            latest = torch.load(output / "latest.pt", weights_only=False)
            self.assertEqual(latest["epoch"], 30)
            torch.testing.assert_close(
                latest["model"]["weight"], torch.full((1, 1), 30.0)
            )


@unittest.skipUnless(
    os.environ.get("UQ_GPU_CHECK") == "1" and torch.cuda.is_available(),
    "Set UQ_GPU_CHECK=1 on a CUDA GPU for full-resolution batch-10 checks.",
)
class FullResolutionGPUTests(unittest.TestCase):
    """Real FP32 models and AdamW states; no reduced test-only backbones."""

    def check_training_batch(self, method, height, width):
        torch.manual_seed(777)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        model = make_model(method, height, width, 17999).cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)
        loader = DataLoader(
            TensorDataset(torch.rand(10, 3, height, width), torch.rand(10, 3, height, width)),
            batch_size=10,
        )
        parameter = next(model.parameters())
        before = parameter.detach().clone()
        result = dict(
            method=method, height=height, width=width, batch_size=10,
            dtype="float32", gpu=torch.cuda.get_device_name(),
            gpu_total_gib=torch.cuda.get_device_properties(0).total_memory / 2**30,
        )
        try:
            # The second step includes already-allocated AdamW moment tensors.
            step = 0
            for _ in range(2):
                loss, step = train_epoch(model, loader, optimizer, "cuda", step, 100)
                self.assertTrue(np.isfinite(loss))
            torch.cuda.synchronize()
            self.assertFalse(torch.equal(parameter.detach(), before))
            self.assertTrue(optimizer.state)
            result.update(
                training_peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                training_peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
            )
            from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

            perceptual = LearnedPerceptualImagePatchSimilarity(
                net_type="alex", normalize=True, reduction="none"
            ).cuda().eval().requires_grad_(False)
            torch.cuda.reset_peak_memory_stats()
            scores = validate(model, loader, perceptual, "cuda")
            torch.cuda.synchronize()
            self.assertTrue(all(np.isfinite(value) for value in scores.values()))
            result.update(
                validation_peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                validation_peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
                scores=scores, passed=True,
            )
        except torch.cuda.OutOfMemoryError:
            result.update(
                passed=False, failure="CUDA out of memory",
                peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
            )
            raise
        finally:
            result["seconds"] = time.monotonic() - started
            print("GPU_CHECK " + json.dumps(result), flush=True)

    def test_qutcc_mirflickr(self):
        self.check_training_batch("qutcc", 270, 480)

    def test_im2im_mirflickr(self):
        self.check_training_batch("im2im", 270, 480)

    def test_ensemble_mirflickr(self):
        self.check_training_batch("ensemble", 270, 480)

    def test_mcdropout_mirflickr(self):
        self.check_training_batch("mcdropout", 270, 480)

    def test_qutcc_diffuser(self):
        self.check_training_batch("qutcc", 300, 480)

    def test_im2im_diffuser(self):
        self.check_training_batch("im2im", 300, 480)

    def test_ensemble_diffuser(self):
        self.check_training_batch("ensemble", 300, 480)

    def test_mcdropout_diffuser(self):
        self.check_training_batch("mcdropout", 300, 480)


if __name__ == "__main__":
    unittest.main(verbosity=2)
