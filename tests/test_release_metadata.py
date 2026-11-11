"""Keep release settings, split indexes and artifact selection coherent."""

import hashlib
import json
from pathlib import Path
import unittest

import numpy as np

from convrml import protocol
from evaluation import download
from convrml.calibration import (
    paper_quantiles,
    risk_threshold,
    rcps_scales,
)

ROOT = Path(__file__).resolve().parents[1]


class ReleaseMetadata(unittest.TestCase):
    def test_all_checkpoint_assets_are_verified_and_uniquely_identified(self):
        artifacts = json.loads(download.INDEX_PATH.read_text())["artifacts"]
        self.assertEqual(len(artifacts), 280)
        self.assertFalse(any(v["dataset"] == "raw_corrupted" for v in artifacts))
        keys = [(v["dataset"], v["method"], v["member"], v["epoch"]) for v in artifacts]
        self.assertEqual(len(keys), len(set(keys)))
        for row in artifacts:
            self.assertTrue(row["github_verified"])
            self.assertRegex(row["sha256"], r"^[a-f0-9]{64}$")
            self.assertGreater(row["size"], 0)
            self.assertEqual(Path(row["name"]).name, row["name"])
        for dataset in ("rml", "diffuser"):
            self.assertEqual(
                [
                    v["member"]
                    for v in download.select_artifacts(dataset, "ensemble", 50)
                ],
                list(range(5)),
            )
            self.assertEqual(
                len(download.select_artifacts(dataset, "ensemble", 50, members=10)), 10
            )

    def test_split_hashes_match_the_published_order_and_have_no_leakage(self):
        historical = json.loads(
            (
                ROOT / "archive/reproducibility/experiment_settings_before_cleanup.json"
            ).read_text()
        )
        for dataset, spec in historical["datasets"].items():
            data = json.loads(
                (
                    ROOT / "archive/reproducibility/splits" / f"{dataset}.json"
                ).read_text()
            )
            seen = set()
            for role, ids in data["ids"].items():
                self.assertEqual(len(ids), spec["counts"][role])
                self.assertEqual(len(ids), len(set(ids)))
                self.assertFalse(seen.intersection(ids))
                seen.update(ids)
                digest = hashlib.sha256(
                    json.dumps(list(map(str, ids)), separators=(",", ":")).encode()
                ).hexdigest()
                self.assertEqual(digest, spec["id_hashes"][role])

    def test_hb_uses_whole_images_and_separate_tail_confidence_budgets(self):
        rng = np.random.default_rng(9)
        scores = rng.uniform(0, 2, (2048, 16)).astype(np.float32)
        fitted = rcps_scales(scores, scores * 2)
        audit = fitted["certificate"]
        self.assertEqual(audit["n_images"], 2048)
        self.assertEqual(audit["tail_delta"], 0.025)
        for side in ("lower", "upper"):
            self.assertGreaterEqual(
                fitted["scales"][side],
                float(
                    np.quantile(
                        scores if side == "lower" else scores * 2, 0.95, method="higher"
                    )
                ),
            )
            self.assertLess(audit["tail_hb_pvalues"][side], 0.025)
        with self.assertRaisesRegex(ValueError, "Insufficient"):
            risk_threshold(2, 0.05, 0.025)

    def test_paper_quantiles_keep_the_reference_finite_image_correction(self):
        result = paper_quantiles(lambda lo, hi: (lo, 1 - hi), 3000)
        self.assertAlmostEqual(result["per_tail_target"], (0.1 - 0.9 / 3000) / 2)
        self.assertLessEqual(result["lower_q"], result["per_tail_target"])
        self.assertLessEqual(1 - result["upper_q"], result["per_tail_target"])
        self.assertNotIn("scales", result)


if __name__ == "__main__":
    unittest.main()
