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

    def test_shared_im2im_preserves_both_legacy_state_names_and_rectangular_outputs(
        self,
    ):
        from models.im2im import Im2im

        class TinyBackbone(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.head = torch.nn.Conv2d(3, 3, 3, padding=1)
                self.up2 = torch.nn.Identity()

            def features(self, y):
                return y

        with patch("models.im2im.ConvNeXtFPN", TinyBackbone):
            torch.manual_seed(13)
            pld = Im2im(checkpoint_prefix="base")
            torch.manual_seed(13)
            dlmd = Im2im(checkpoint_prefix="net")
        self.assertIs(type(pld), type(dlmd))
        pld_state = {
            key.removeprefix("base."): value for key, value in pld.state_dict().items()
        }
        dlmd_state = {
            key.removeprefix("net."): value for key, value in dlmd.state_dict().items()
        }
        self.assertEqual(pld_state.keys(), dlmd_state.keys())
        for key in pld_state:
            self.assertTrue(torch.equal(pld_state[key], dlmd_state[key]))
        y = torch.rand(2, 3, 33, 47)
        torch.testing.assert_close(pld(y), dlmd(y), rtol=0, atol=0)
        self.assertEqual(pld(y).shape, (2, 3, 3, 33, 47))


if __name__ == "__main__":
    unittest.main()
