"""Check the publication API, paired-file conventions and native intervals."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dataset import ImagePairs, NumpyPairs


class PairedData(unittest.TestCase):
    def test_numpy_split_order_and_separate_calibration_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pairs = np.stack(
                [np.full((2, 3, 4, 5), value, dtype=np.uint8) for value in (10, 20, 30)]
            )
            np.save(root / "pairs.npy", pairs)
            manifest = {role: [dict(id=role)] for role in ("train", "val", "test")}
            (root / "manifest.json").write_text(json.dumps(manifest))
            for role, value in zip(manifest, pairs):
                dataset = NumpyPairs(root / "pairs.npy", root / "manifest.json", role)
                self.assertEqual(dataset.ids, (role,))
                torch.testing.assert_close(
                    dataset[0], torch.from_numpy(value).float() / 255, rtol=0, atol=0
                )
                torch.testing.assert_close(dataset[-1], dataset[0], rtol=0, atol=0)
                with self.assertRaises(IndexError):
                    dataset[len(dataset)]
            with self.assertRaises(ValueError):
                NumpyPairs(root / "pairs.npy", root / "manifest.json", "calibration")
            np.save(root / "calibration.npy", pairs[:1])
            (root / "calibration.json").write_text(
                json.dumps({"calibration": [{"id": "heldout"}]})
            )
            dataset = NumpyPairs(
                root / "calibration.npy", root / "calibration.json", "calibration"
            )
            self.assertEqual(dataset.ids, ("heldout",))

    def test_image_pairs_keep_rectangular_dimensions_rgb_values_and_target(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            y = np.full((7, 11, 3), 80, dtype=np.uint8)
            x = np.full((7, 11, 3), 160, dtype=np.uint8)
            Image.fromarray(y).save(root / "measurement.png")
            Image.fromarray(x).save(root / "target.png")
            row = dict(
                id="scene",
                measurement=dict(path="measurement.png"),
                target=dict(path=str(root / "target.png")),
            )
            (root / "split.json").write_text(json.dumps({"roles": {"test": [row]}}))
            dataset = ImagePairs(root / "split.json", "test")
            actual = dataset[0]
            self.assertEqual(actual.shape, (2, 3, 7, 11))
            self.assertEqual(dataset.ids, ("scene",))
            expected = torch.stack(
                [
                    torch.from_numpy(value).permute(2, 0, 1).float() / 255
                    for value in (y, x)
                ]
            )
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_models_are_defined_once_without_dataset_branches(self):
        root = ROOT / "models"
        self.assertEqual(
            {p.stem for p in root.glob("*.py") if p.stem != "__init__"},
            {"im2im", "ensemble", "mc_dropout", "qutcc"},
        )
        self.assertFalse(any(root.glob("*/im2im.py")))
        self.assertFalse(any(root.glob("*/qutcc.py")))

    def test_im2im_has_one_namespace_and_loads_legacy_keys_for_both_datasets(self):
        from collections import OrderedDict
        from convrml.backbone import ConvNeXtFNN
        from convrml.checkpoints import load_model, load_model_weights
        from convrml.protocol import profile
        from models.im2im import Im2im

        class TinyBackbone(ConvNeXtFNN):
            def __init__(self, *, output_heads=1):
                torch.nn.Module.__init__(self)
                self.output_heads = output_heads
                self.head = torch.nn.Conv2d(3, 3 * output_heads, 3, padding=1)
                self.up2 = torch.nn.Identity()

            def features(self, y):
                return y

        with (
            patch("models.im2im.ConvNeXtFNN", TinyBackbone),
            tempfile.TemporaryDirectory() as directory,
        ):
            reference = Im2im()
            canonical = reference.state_dict()
            self.assertTrue(all(key.startswith("backbone.") for key in canonical))
            self.assertEqual(reference.backbone.head.out_channels, 9)
            image = torch.rand(2, 3, 33, 47)
            expected = reference(image)
            self.assertEqual(expected.shape, (2, 3, 3, 33, 47))
            for root in ("base", "net", "backbone"):
                legacy = OrderedDict(
                    (key.replace("backbone.", root + ".", 1), value)
                    for key, value in canonical.items()
                )
                legacy._metadata = {"": {"version": 1}, root: {"version": 7}}
                original_keys = list(legacy)
                direct = Im2im()
                metadata_versions = []
                handle = direct.backbone.register_load_state_dict_pre_hook(
                    lambda module, state, prefix, metadata, *args: metadata_versions.append(
                        metadata["version"]
                    )
                )
                load_model_weights(direct, legacy)
                handle.remove()
                self.assertEqual(metadata_versions, [7])
                self.assertEqual(list(legacy), original_keys)
                self.assertIn(root, legacy._metadata)
                torch.testing.assert_close(direct(image), expected, rtol=0, atol=0)
                for experiment, dataset in ((2, "rml"), (3, "raw_clean")):
                    with self.subTest(prefix=root, dataset=dataset):
                        run = profile(experiment, dataset, "im2im")
                        path = Path(directory) / "weights.pt"
                        torch.save(dict(run=run, epoch=5, model=legacy), path)
                        loaded, _ = load_model(path, run, torch.device("cpu"))
                        torch.testing.assert_close(
                            loaded(image), expected, rtol=0, atol=0
                        )
                        self.assertEqual(list(loaded.state_dict()), list(canonical))
                        self.assertEqual(list(legacy), original_keys)
                        self.assertIn(root, legacy._metadata)
                mixed = OrderedDict(legacy)
                mixed["other.head.weight"] = reference.backbone.head.weight
                with self.assertRaises((ValueError, RuntimeError)):
                    load_model_weights(Im2im(), mixed)
            incomplete = dict(canonical)
            del incomplete["backbone.head.weight"]
            with self.assertRaises(RuntimeError):
                load_model_weights(Im2im(), incomplete)
            wrong_shape = dict(canonical)
            wrong_shape["backbone.head.weight"] = reference.backbone.head.weight[:3]
            with self.assertRaises(RuntimeError):
                load_model_weights(Im2im(), wrong_shape)

    def test_backbone_pads_rectangles_and_crops_without_breaking_gradients(self):
        from convrml.backbone import ConvNeXtFNN

        class SpatialBackbone(ConvNeXtFNN):
            def __init__(self):
                torch.nn.Module.__init__(self)
                self.output_heads = 1
                self.head = torch.nn.Conv2d(3, 3, 3, padding=1)
                self.up2 = torch.nn.Identity()

            def features(self, image):
                self.padded_shape = image.shape[-2:]
                self.assert_aligned = all(size % 32 == 0 for size in self.padded_shape)
                return image

        model = SpatialBackbone()
        for height, width in ((32, 64), (33, 47), (270, 480)):
            image = torch.rand(1, 3, height, width, requires_grad=True)
            output = model(image)
            self.assertTrue(model.assert_aligned)
            self.assertEqual(
                model.padded_shape, ((height + 31) // 32 * 32, (width + 31) // 32 * 32)
            )
            self.assertEqual(output.shape, image.shape)
            output.square().mean().backward()
            self.assertTrue(torch.isfinite(image.grad).all())

    def test_gaussian_member_crops_mean_variance_and_all_ensemble_bounds(self):
        from models.ensemble import _GaussianBackbone, EnsembleMember, DeepEnsemble

        class SpatialGaussian(_GaussianBackbone):
            def __init__(self):
                torch.nn.Module.__init__(self)
                self.output_heads = 1
                self.head = torch.nn.Conv2d(3, 3, 3, padding=1)
                self.variance_head = torch.nn.Conv2d(3, 3, 3, padding=1)
                self.up2 = torch.nn.Identity()

            def features(self, image):
                if any(size % 32 for size in image.shape[-2:]):
                    raise ValueError("Encoder requires stride-aligned input")
                return image

        member = EnsembleMember(base_model=SpatialGaussian())
        ensemble = DeepEnsemble([member])
        for height, width in ((32, 64), (33, 47), (270, 480)):
            image = torch.rand(1, 3, height, width)
            mean, variance = member(image)
            self.assertEqual(mean.shape, image.shape)
            self.assertEqual(variance.shape, image.shape)
            self.assertTrue((variance > 0).all())
            for endpoint in ensemble.bounds(image):
                self.assertEqual(endpoint.shape, image.shape)
            self.assertEqual(ensemble(image).shape, (1, 9, height, width))


if __name__ == "__main__":
    unittest.main()
