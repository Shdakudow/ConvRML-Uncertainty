"""Protect the single-source layout and experiment-specific entry-point paths."""

import hashlib
import importlib.util
import json
from pathlib import Path
import unittest
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


class SharedSourceLayout(unittest.TestCase):
    def test_aliases_resolve_to_the_recorded_implementation(self):
        record = json.loads((ROOT / "shared/source_map.json").read_text())
        self.assertGreaterEqual(len(record["aliases"]), 40)
        for alias, implementation in record["aliases"].items():
            with self.subTest(alias=alias):
                source = ROOT / alias
                self.assertTrue(source.is_symlink())
                self.assertEqual(source.resolve(), ROOT / implementation)
        for implementation, digest in record["sha256"].items():
            actual = hashlib.sha256((ROOT / implementation).read_bytes()).hexdigest()
            self.assertEqual(actual, digest, implementation)

    def test_experiment_paths_are_distinct_even_when_code_is_shared(self):
        exp2 = ROOT / "experiments/experiment2/four_models/qutcc/train.py"
        exp3 = ROOT / "experiments/experiment3/clean/train.py"
        self.assertEqual(exp2.resolve(), exp3.resolve())
        self.assertNotEqual(exp2.absolute().parent, exp3.absolute().parent)
        self.assertIn("ROOT = Path(__file__).absolute().parent", exp2.read_text())

    @unittest.skipUnless(importlib.util.find_spec("torch"), "ML environment required")
    def test_cli_aliases_and_output_roots(self):
        folders = [
            ROOT / "experiments/experiment2/four_models/qutcc",
            ROOT / "experiments/experiment3/clean",
        ]
        with tempfile.TemporaryDirectory() as elsewhere:
            for folder in folders:
                for script in ["train.py", "evaluate.py"]:
                    output = subprocess.check_output(
                        [sys.executable, str(folder / script), "--help"],
                        cwd=elsewhere,
                        text=True,
                    )
                    self.assertIn("--task", output)
                code = (
                    "import sys;sys.path.insert(0," + repr(str(folder)) + ");"
                    "import train;print(train.ROOT);"
                    "assert train.ROOT == __import__('pathlib').Path("
                    + repr(str(folder))
                    + ");"
                    "assert any(p.is_symlink() for p in train.ROOT.rglob('*.py'));"
                    "print(train.source_hash())"
                )
                output = subprocess.check_output(
                    [sys.executable, "-c", code], cwd=elsewhere, text=True
                )
                self.assertEqual(output.splitlines()[0], str(folder))
                self.assertEqual(len(output.splitlines()[1]), 64)


if __name__ == "__main__":
    unittest.main()
