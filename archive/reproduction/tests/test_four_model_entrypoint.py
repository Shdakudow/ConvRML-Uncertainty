"""Verify new QUTCC routing, independent members, and historical-source isolation."""

import hashlib
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "experiments/experiment2"
SPEC = importlib.util.spec_from_file_location("entrypoint", BASE / "run_model.py")
entrypoint = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(entrypoint)


class FourModelEntrypoint(unittest.TestCase):
    def build(self, *args):
        parsed = entrypoint.arguments(
            [*args, "--project-root", "/tmp/convrml-test-project"]
        )
        return entrypoint.command(parsed)

    def test_current_qutcc_train_and_calibration(self):
        cmd, cwd, env = self.build(
            "train", "--method", "qutcc", "--camera", "diffuser", "--epoch", "5"
        )
        self.assertEqual(Path(cmd[1]), BASE / "qutcc/train.py")
        self.assertEqual(cmd[2:], ["--task", "1", "--target-epoch", "5"])
        self.assertEqual(cwd, BASE / "qutcc")
        self.assertEqual(
            env["CONVRML_PROJECT"], str(Path("/tmp/convrml-test-project").resolve())
        )
        cmd, _, _ = self.build(
            "evaluate", "--method", "qutcc", "--camera", "rml", "--epoch", "50"
        )
        self.assertEqual(Path(cmd[1]), BASE / "qutcc/evaluate.py")
        self.assertNotIn("conditioned", cmd)
        self.assertNotIn("adaptive_qutcc", cmd)

    def test_mc_and_ensemble_keep_baseline_protocol(self):
        cmd, cwd, _ = self.build(
            "evaluate", "--method", "mc_dropout", "--camera", "rml", "--epoch", "10"
        )
        self.assertEqual(cmd[cmd.index("--method") + 1], "mc_005")
        self.assertEqual(cwd, Path("/tmp/convrml-test-project").resolve())
        cmd, _, _ = self.build(
            "train",
            "--method",
            "ensemble",
            "--member",
            "4",
            "--camera",
            "rml",
            "--epoch",
            "2",
        )
        self.assertEqual(cmd[cmd.index("--arm") + 1], "ensemble_04")
        cmd, _, _ = self.build(
            "evaluate",
            "--method",
            "ensemble",
            "--members",
            "10",
            "--camera",
            "diffuser",
            "--epoch",
            "50",
        )
        self.assertEqual(cmd[cmd.index("--method") + 1], "ensemble10")

    def test_tight_coverage_is_separate_from_reference_and_training(self):
        cmd, _, _ = self.build(
            "evaluate",
            "--method",
            "qutcc",
            "--camera",
            "rml",
            "--epoch",
            "50",
            "--calibration",
            "tight90",
            "--calibration-source",
            "/tmp/paper-source",
        )
        self.assertEqual(Path(cmd[1]), ROOT / "tools/evaluate_coverage90.py")
        self.assertEqual(cmd[cmd.index("--family") + 1], "paper")
        self.assertEqual(
            cmd[cmd.index("--source-folder") + 1],
            str(Path("/tmp/paper-source").resolve()),
        )
        self.assertIn("tight_coverage90", cmd[-1])
        with self.assertRaises(SystemExit):
            self.build(
                "train",
                "--method",
                "im2im",
                "--camera",
                "rml",
                "--calibration",
                "tight90",
            )

    def test_qutcc_copy_is_exact_current_source(self):
        metadata = json.loads((BASE / "qutcc/provenance.json").read_text())
        for relative, digest in metadata["sha256"].items():
            with self.subTest(file=relative):
                actual = (BASE / "qutcc" / relative).read_bytes()
                self.assertEqual(hashlib.sha256(actual).hexdigest(), digest)
                self.assertEqual(
                    actual, (ROOT / metadata["source"] / relative).read_bytes()
                )


if __name__ == "__main__":
    unittest.main()
