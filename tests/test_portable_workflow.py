"""Exercise portable training/resume, split isolation, checkpoints and evaluation."""

import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from torch import nn

import train
import calibration as evaluate
from convrml import checkpoints, download, protocol
from models.qutcc import pinball
from models import Im2im, QUTCC


class TinyQ(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.rand(()) * 0.1 + 0.8)
        self.dropout = nn.Dropout(0.1)

    def forward(self, y, q=None):
        q = y.new_full((len(y),), 0.5) if q is None else q
        return self.dropout(y * self.scale) + (q[:, None, None, None] - 0.5) * 0.2

    def center(self, y):
        return self(y)

    def loss(self, y, x, *, generator=None):
        q = torch.rand(len(y), device=y.device, generator=generator)
        return pinball(self(y, q), x, q)

    def bounds(self, y, lower_q=0.05, upper_q=0.95):
        return (
            self(y),
            self(y, y.new_full((len(y),), lower_q)),
            self(y, y.new_full((len(y),), upper_q)),
        )


class TinyIntervals(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.5))

    def bounds(self, y):
        center = y * 0 + self.scale
        return center, center - 0.1, center + 0.1


class PortableWorkflow(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        rng = np.random.default_rng(42)
        pairs = rng.integers(0, 256, (8, 2, 3, 16, 16), dtype=np.uint8)
        np.save(self.root / "pairs.npy", pairs)
        manifest = {
            role: [dict(id=f"{role}{i}") for i in range(count)]
            for role, count in [("train", 4), ("val", 2), ("test", 2)]
        }
        (self.root / "manifest.json").write_text(json.dumps(manifest))
        np.save(
            self.root / "calibration.npy",
            rng.integers(0, 256, (128, 2, 3, 16, 16), dtype=np.uint8),
        )
        (self.root / "calibration.json").write_text(
            json.dumps({"calibration": [dict(id=f"cal{i}") for i in range(128)]})
        )
        self.config = protocol.settings()
        self.config["datasets"]["rml"].update(
            counts=dict(train=4, val=2, test=2, calibration=128),
            shape=[3, 16, 16],
            id_hashes={},
        )
        patcher = patch.object(protocol, "settings", return_value=self.config)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_new_profiles_use_adamw_with_explicit_decay(self):
        for dataset, data in self.config["datasets"].items():
            experiment = data["experiment"]
            for method in self.config["methods"][f"experiment{experiment}"]:
                member = 0 if method == "ensemble" else None
                config = protocol.profile(experiment, dataset, method, member)
                optimizer = train.create_optimizer(TinyQ(), config, 100)
                self.assertIsInstance(optimizer, torch.optim.AdamW)
                self.assertEqual(
                    optimizer.param_groups[0]["weight_decay"], config["weight_decay"]
                )
                self.assertEqual(
                    optimizer.param_groups[0]["betas"], tuple(config["betas"])
                )
        historical = dict(config, optimizer="Adam")
        with self.assertRaisesRegex(ValueError, "historical Adam"):
            train.create_optimizer(TinyQ(), historical, 100)

    def training_args(self, output, epoch, resume=False):
        values = [
            "--dataset",
            "rml",
            "--method",
            "qutcc",
            "--manifest",
            str(self.root / "manifest.json"),
            "--pairs",
            str(self.root / "pairs.npy"),
            "--output",
            str(self.root / output),
            "--epochs",
            str(epoch),
            "--workers",
            "0",
            "--device",
            "cpu",
        ]
        return train.arguments(values + (["--resume"] if resume else []))

    def factory(self, run):
        torch.manual_seed(run["seeds"]["initialization"])
        return TinyQ()

    def evaluation_args(self, output="evaluation", method="im2im"):
        return evaluate.arguments(
            [
                "--dataset",
                "rml",
                "--method",
                method,
                "--checkpoints",
                "fixture.pt",
                "--manifest",
                str(self.root / "manifest.json"),
                "--pairs",
                str(self.root / "pairs.npy"),
                "--calibration-manifest",
                str(self.root / "calibration.json"),
                "--calibration-pairs",
                str(self.root / "calibration.npy"),
                "--output",
                str(self.root / output),
                "--workers",
                "0",
                "--device",
                "cpu",
                "--examples",
                "1",
                "--no-lpips",
            ]
        )

    def model_loader(self, path, run, device, **kwargs):
        return TinyIntervals().to(device).eval(), dict(
            epoch=50, seeds={}, file="fixture.pt", sha256="0" * 64
        )

    def test_dataset_selects_its_recorded_profile(self):
        for dataset, expected in (
            ("rml", 2),
            ("diffuser", 2),
            ("raw_clean", 3),
            ("raw_corrupted", 3),
        ):
            common = [
                "--dataset",
                dataset,
                "--method",
                "qutcc",
                "--manifest",
                "manifest.json",
                "--output",
                "unused",
            ]
            for module, extra in (
                (train, []),
                (evaluate, ["--checkpoints", "weights.pt"]),
            ):
                args = module.arguments(common + extra)
                self.assertEqual(args.experiment, expected)
                wrong = 3 if expected == 2 else 2
                with self.assertRaisesRegex(ValueError, "Dataset does not belong"):
                    protocol.profile(wrong, dataset, "qutcc")

    def test_resume_matches_uninterrupted_parameters_rng_and_loss(self):
        direct = train.run_training(
            self.training_args("direct", 2), factory=self.factory
        )
        train.run_training(self.training_args("resumed", 1), factory=self.factory)
        resumed = train.run_training(
            self.training_args("resumed", 2, True), factory=self.factory
        )
        a, _ = checkpoints.read_checkpoint(direct)
        b, _ = checkpoints.read_checkpoint(resumed)
        for name in a["model"]:
            self.assertTrue(torch.equal(a["model"][name], b["model"][name]), name)
        for name in ("rng", "quantile_rng", "data_rng"):
            self.assertTrue(torch.equal(a[name], b[name]), name)
        self.assertEqual(
            [r["loss"] for r in a["history"]], [r["loss"] for r in b["history"]]
        )
        self.assertEqual(a["global_step"], b["global_step"])

    def test_custom_factory_initialization_ignores_ambient_rng(self):
        # Factories need not seed themselves: setup owns that contract.
        paths = []
        for ambient_seed in (19, 82):
            torch.manual_seed(ambient_seed)
            paths.append(
                train.run_training(
                    self.training_args(f"seed_{ambient_seed}", 1),
                    factory=lambda run: TinyQ(),
                )
            )
        first, _ = checkpoints.read_checkpoint(paths[0])
        second, _ = checkpoints.read_checkpoint(paths[1])
        for key in first["model"]:
            self.assertTrue(torch.equal(first["model"][key], second["model"][key]))
        self.assertEqual(first["history"][0]["loss"], second["history"][0]["loss"])

    def test_dlmd_image_files_train_load_calibrate_and_export_both_methods(self):
        """Exercise the paired-file workflow with tiny models and real loss methods."""

        class QuantileFixture(nn.Module):
            loss = QUTCC.loss
            bounds = QUTCC.bounds
            center = QUTCC.center

            def __init__(self):
                super().__init__()
                self.bias = nn.Parameter(torch.zeros(()))

            def forward(self, y, q=None):
                q = y.new_full((len(y),), 0.5) if q is None else q
                return y * 0 + q[:, None, None, None] + self.bias

        class Im2imFixture(nn.Module):
            loss = Im2im.loss
            bounds = Im2im.bounds
            center = Im2im.center

            def __init__(self):
                super().__init__()
                self.heads = nn.Parameter(torch.tensor([0.05, 0.5, 0.95]))

            def forward(self, y):
                return y[:, None] * 0 + self.heads[None, :, None, None, None]

        roles = {}
        rng = np.random.default_rng(942)
        for role, count in (
            ("train", 4),
            ("val", 2),
            ("test", 2),
            ("calibration", 128),
        ):
            roles[role] = []
            for index in range(count):
                row = dict(id=f"{role}{index}")
                for field in ("measurement", "target"):
                    filename = f"{role}_{index}_{field}.png"
                    pixels = rng.integers(0, 256, (16, 16, 3), dtype=np.uint8)
                    Image.fromarray(pixels).save(self.root / filename)
                    row[field] = dict(path=filename)
                roles[role].append(row)
        manifest = self.root / "raw.json"
        manifest.write_text(json.dumps(dict(roles=roles)))
        self.config["datasets"]["raw_clean"].update(
            counts={role: len(rows) for role, rows in roles.items()},
            shape=[3, 16, 16],
            id_hashes={},
        )

        def factory(run):
            return QuantileFixture() if run["method"] == "qutcc" else Im2imFixture()

        def load_model(path, run, device, **kwargs):
            return checkpoints.load_model(path, run, device, factory=factory, **kwargs)

        for method in ("im2im", "qutcc"):
            with self.subTest(method=method):
                args = self.training_args(f"raw_{method}", 2)
                args.dataset, args.experiment, args.method = "raw_clean", 3, method
                args.manifest, args.pairs = manifest, None
                checkpoint = train.run_training(args, factory=factory)
                args = self.evaluation_args(f"raw_{method}_test", method)
                args.dataset, args.experiment = "raw_clean", 3
                args.manifest = args.calibration_manifest = manifest
                args.pairs = args.calibration_pairs = None
                args.checkpoints = [checkpoint]
                report = evaluate.run_evaluation(args, model_loader=load_model)
                self.assertEqual(report["test_count"], 2)
                self.assertEqual(report["epoch"], 2)
                self.assertEqual(report["calibration"]["n_images"], 128)
                self.assertFalse(report["request"]["test_used_for_calibration"])
                self.assertEqual(
                    report["calibration"]["has_hb_certificate"], method == "im2im"
                )
                self.assertTrue((args.output / "calibration.json").is_file())
                self.assertTrue((args.output / "per_image.csv").is_file())
                with np.load(args.output / "examples.npz") as examples:
                    for name in (
                        "measurement",
                        "target",
                        "prediction",
                        "lower",
                        "upper",
                    ):
                        self.assertEqual(examples[name].shape, (1, 3, 16, 16))

    def test_five_epoch_snapshot_contains_weights_and_epoch(self):
        latest = train.run_training(
            self.training_args("milestones", 5), factory=self.factory
        )
        snapshot, _ = checkpoints.read_checkpoint(
            latest.parent / "checkpoints/epoch_005.pt"
        )
        full, _ = checkpoints.read_checkpoint(latest)
        for key in full["model"]:
            torch.testing.assert_close(
                snapshot["model"][key], full["model"][key], rtol=0, atol=0
            )
        self.assertEqual(snapshot["epoch"], 5)
        self.assertNotIn("optimizer", snapshot)

    def test_baseline_resume_keeps_persistent_shuffle_and_dropout_rng(self):
        class TinyMC(TinyQ):
            def loss(self, y, target):
                return (self(y) - target).square().mean()

        def factory(run):
            torch.manual_seed(run["seeds"]["initialization"])
            return TinyMC()

        def args(folder, epoch, resume=False):
            result = self.training_args(folder, epoch, resume)
            result.method = "mc_dropout"
            return result

        direct = train.run_training(args("mc_direct", 2), factory=factory)
        train.run_training(args("mc_resumed", 1), factory=factory)
        resumed = train.run_training(args("mc_resumed", 2, True), factory=factory)
        a, _ = checkpoints.read_checkpoint(direct)
        b, _ = checkpoints.read_checkpoint(resumed)
        for key in a["model"]:
            self.assertTrue(torch.equal(a["model"][key], b["model"][key]))
        for key in ("rng", "data_rng"):
            self.assertTrue(torch.equal(a[key], b[key]))
        self.assertEqual(
            [row["loss"] for row in a["history"]], [row["loss"] for row in b["history"]]
        )

    def test_resume_refuses_changed_training_settings(self):
        latest = train.run_training(
            self.training_args("guarded", 1), factory=self.factory
        )
        checkpoint, _ = checkpoints.read_checkpoint(latest)
        checkpoint["run"]["batch_size"] += 1
        checkpoints.save_checkpoint(latest, checkpoint)
        with self.assertRaisesRegex(ValueError, "settings or dataset changed"):
            train.run_training(
                self.training_args("guarded", 2, True), factory=self.factory
            )

    def test_evaluation_fits_only_calibration_then_exports_exact_test_ids(self):
        args = self.evaluation_args()
        original = evaluate.dataset_for

        def check_freeze(dataset, manifest, pairs, role):
            if role == "test":
                self.assertTrue((args.output / "calibration.json").exists())
            return original(dataset, manifest, pairs, role)

        with patch.object(evaluate, "dataset_for", side_effect=check_freeze):
            report = evaluate.run_evaluation(args, model_loader=self.model_loader)
        self.assertEqual(report["test_count"], 2)
        self.assertEqual([row["id"] for row in report["per_image"]], ["test0", "test1"])
        self.assertFalse(report["request"]["test_used_for_calibration"])
        self.assertEqual(report["calibration"]["n_images"], 128)
        self.assertTrue(report["calibration"]["has_hb_certificate"])
        self.assertEqual(report["request"]["calibration"], "paper")
        self.assertEqual(report["calibration"]["procedure"], "asymmetric RCPS-HB")
        for mode in ("raw", "clipped"):
            for key, average in report["metrics"][mode].items():
                values = [
                    row[mode][key]
                    for row in report["per_image"]
                    if row[mode][key] is not None
                ]
                self.assertEqual(average, sum(values) / len(values) if values else None)
                self.assertEqual(report["metric_valid_counts"][mode][key], len(values))
        self.assertIsNone(report["metrics"]["clipped"]["lpips"])
        self.assertIsNone(report["metrics"]["clipped"]["pearson"])
        arrays = np.load(args.output / "examples.npz")
        expected_target = np.load(self.root / "pairs.npy")[6, 1] / 255
        np.testing.assert_allclose(arrays["target"][0], expected_target, atol=3e-8)
        self.assertTrue((args.output / "calibration.json").exists())
        with self.assertRaises(FileExistsError):
            evaluate.run_evaluation(args, model_loader=self.model_loader)

    def test_calibration_overlap_is_rejected_even_in_smoke_mode(self):
        path = self.root / "calibration.json"
        path.write_text(
            json.dumps(
                {
                    "calibration": [dict(id="test0")]
                    + [dict(id=f"cal{i}") for i in range(1, 128)]
                }
            )
        )
        args = self.evaluation_args("overlap")
        args.smoke = True
        with self.assertRaisesRegex(ValueError, "overlap"):
            evaluate.run_evaluation(args, model_loader=self.model_loader)

    def test_infeasible_quantiles_never_load_test_pixels_or_report_zero(self):
        class Impossible(TinyQ):
            def forward(self, y, q=None):
                q = y.new_full((len(y),), 0.5) if q is None else q
                return torch.zeros_like(y) + 0.5 + q[:, None, None, None] * 0.1

        def loader(*args, **kwargs):
            return Impossible().eval(), dict(
                epoch=20, seeds={}, file="fixture.pt", sha256="0" * 64
            )

        args = self.evaluation_args("infeasible", "qutcc")
        roles = []
        original = evaluate.dataset_for

        def record(dataset, manifest, pairs, role):
            roles.append(role)
            return original(dataset, manifest, pairs, role)

        with patch.object(evaluate, "dataset_for", side_effect=record):
            with self.assertRaises(evaluate.CalibrationFailure):
                evaluate.run_evaluation(args, model_loader=loader)
        self.assertNotIn("test", roles)
        self.assertTrue((args.output / "failure.json").exists())
        self.assertFalse((args.output / "report.json").exists())

    def test_checkpoint_identity_and_strict_keys_are_checked(self):
        run = protocol.profile(2, "rml", "qutcc")
        path = self.root / "weights.pt"
        checkpoints.save_checkpoint(
            path, dict(run=run, epoch=5, model=self.factory(run).state_dict())
        )
        model, info = checkpoints.load_model(
            path, run, torch.device("cpu"), factory=self.factory
        )
        self.assertEqual(info["sha256"], protocol.sha256(path))
        wrong = dict(run, method="im2im")
        with self.assertRaisesRegex(ValueError, "method"):
            checkpoints.load_model(
                path, wrong, torch.device("cpu"), factory=self.factory
            )

    def test_removed_calibration_modes_are_rejected_without_loading_weights(self):
        for mode in ("empirical", "reference"):
            args = self.evaluation_args("removed_" + mode)
            args.calibration = mode
            with self.assertRaisesRegex(ValueError, "paper or native"):
                evaluate.run_evaluation(args, model_loader=self.model_loader)

    def test_native_control_does_not_open_calibration_pixels(self):
        args = self.evaluation_args("native")
        args.calibration = "native"
        args.calibration_manifest = None
        args.calibration_pairs = None
        original, roles = evaluate.dataset_for, []

        def record(dataset, manifest, pairs, role):
            roles.append(role)
            return original(dataset, manifest, pairs, role)

        with patch.object(evaluate, "dataset_for", side_effect=record):
            report = evaluate.run_evaluation(args, model_loader=self.model_loader)
        self.assertEqual(roles, ["test"])
        self.assertEqual(report["request"]["calibration"], "native")
        self.assertEqual(report["calibration"], {})

    def test_lpips_export_is_per_image_clipped_only_and_counted(self):
        class FixtureLPIPS:
            def __init__(self, device):
                pass

            def __call__(self, prediction, target):
                return torch.full((len(prediction),), 0.125)

        args = self.evaluation_args("with_lpips")
        args.no_lpips = False
        with patch.object(evaluate, "PerceptualMetric", FixtureLPIPS):
            report = evaluate.run_evaluation(args, model_loader=self.model_loader)
        self.assertEqual(report["metrics"]["clipped"]["lpips"], 0.125)
        self.assertEqual(report["metric_valid_counts"]["clipped"]["lpips"], 2)
        self.assertIsNone(report["metrics"]["raw"]["lpips"])
        self.assertEqual(report["metric_valid_counts"]["raw"]["lpips"], 0)
        self.assertIn("clipped_lpips", (args.output / "per_image.csv").read_text())
        self.assertIn("prediction_seconds_per_image", report["timing"])
        self.assertIn("metric_seconds", report["timing"])

    def test_split_isolation_and_independent_ensemble_seeds(self):
        path = self.root / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["test"][0]["id"] = manifest["train"][0]["id"]
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "overlapping"):
            protocol.audit_manifest(path)
        seeds = [
            protocol.profile(2, "rml", "ensemble", member)["seeds"]
            for member in range(5)
        ]
        self.assertEqual(len({s["initialization"] for s in seeds}), 5)
        self.assertEqual(len({s["data"] for s in seeds}), 5)

    def test_verified_download_reuses_bytes_and_rejects_corruption(self):
        body = b"verified checkpoint"
        source = self.root / "source.pt"
        source.write_bytes(body)
        entry = dict(
            name="download.pt",
            size=len(body),
            sha256=protocol.sha256(source),
            url="https://example.invalid/file",
            release="fixture",
        )
        with (
            patch.object(download.shutil, "which", return_value=None),
            patch.object(
                download.urllib.request, "urlopen", return_value=io.BytesIO(body)
            ),
        ):
            target = download.download_one(entry, self.root / "downloads")
        self.assertEqual(download.download_one(entry, self.root / "downloads"), target)
        target.write_bytes(b"changed")
        with self.assertRaises(FileExistsError):
            download.download_one(entry, self.root / "downloads")


if __name__ == "__main__":
    unittest.main()
