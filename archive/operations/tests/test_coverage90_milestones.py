"""Failed calibration is not a fabricated test point or an infrastructure retry."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RUNNER = ROOT / "run_coverage90_milestone.py"


class Milestones(unittest.TestCase):
    def run_case(self, folder, command):
        path = folder / "tasks.json"
        path.write_text(
            json.dumps(
                [
                    dict(
                        index=0,
                        group="rml",
                        method="qutcc",
                        epoch=5,
                        output=str(folder / "out"),
                        command=command,
                    )
                ]
            )
        )
        return subprocess.run(
            [sys.executable, str(RUNNER), str(path), "0"],
            capture_output=True,
            text=True,
        )

    def test_known_infeasibility_is_a_gap_without_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            result = self.run_case(
                folder,
                [
                    sys.executable,
                    "-c",
                    "import json,pathlib;pathlib.Path("
                    + repr(str(folder / "out/failure.json"))
                    + ").write_text(json.dumps(dict(status='calibration_infeasible',group='rml',method='qutcc',epoch=5,error_code='infeasible_bounds',error='No feasible extreme quantile endpoints',test_used_for_calibration=False)));raise SystemExit(2)",
                ],
            )
            self.assertEqual(result.returncode, 0)
            error = json.loads((folder / "out/failure.json").read_text())
            self.assertEqual(error["status"], "calibration_infeasible")
            self.assertNotIn("metrics", error)
            self.assertFalse((folder / "out/report.json").exists())

    def test_execution_error_is_retryable(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            result = self.run_case(
                folder, [sys.executable, "-c", "raise SystemExit(42)"]
            )
            self.assertEqual(result.returncode, 42)
            self.assertEqual(
                json.loads((folder / "out/failure.json").read_text())["status"],
                "execution_failed",
            )

    def test_matching_log_phrase_does_not_fake_calibration_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            result = self.run_case(
                folder,
                [
                    sys.executable,
                    "-c",
                    "print('No feasible extreme quantile endpoints');raise SystemExit(42)",
                ],
            )
            self.assertEqual(result.returncode, 42)
            self.assertEqual(
                json.loads((folder / "out/failure.json").read_text())["status"],
                "execution_failed",
            )

    def test_every_ensemble_member_must_exist_in_archive_manifest(self):
        spec = importlib.util.spec_from_file_location(
            "milestone_dispatch", ROOT / "dispatch_coverage90_milestones.py"
        )
        dispatch = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(dispatch)
        with tempfile.TemporaryDirectory() as directory:
            dispatch.ARCHIVE = Path(directory)
            for camera in ("rml", "diffuser"):
                rows = [
                    dict(path=f"{camera}/{arm}/checkpoints/epoch_005.pt")
                    for arm in dispatch.ARMS
                ]
                (dispatch.ARCHIVE / (camera + "-manifest.json")).write_text(
                    json.dumps(dict(release="fixture", files=rows))
                )
            self.assertEqual(len(dispatch.entries(5)), 14)
            path = dispatch.ARCHIVE / "rml-manifest.json"
            data = json.loads(path.read_text())
            data["files"].pop()
            path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                dispatch.entries(5)


if __name__ == "__main__":
    unittest.main()
