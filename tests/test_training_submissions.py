"""Exercise Slurm submission plans without submitting real GPU jobs."""

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1] / "experiments/generalized_inputs"


def load_script(name):
    spec = importlib.util.spec_from_file_location(
        "generalized_" + name, SOURCE / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(SOURCE))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


submit = load_script("submit")
schedule = load_script("schedule_after50")
slurm = sys.modules["slurm"]


class SubmissionPlans(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.repository = self.root / "repository"
        self.source = self.repository / "experiments/generalized_inputs"
        self.source.mkdir(parents=True)
        self.run = self.root / "run"
        self.project = self.root / "project"
        self.manifest = (
            self.project / "experiments/dlmd_raw_adaptive_size_20261016/data/split.json"
        )
        self.manifest.parent.mkdir(parents=True)
        self.manifest.write_text("{}")
        for name in ("train.py", "run.sbatch", "continue_after50.py"):
            (self.source / name).write_text("fixture")
        (self.repository / "source_commit.txt").write_text("frozen-test-revision")
        (self.source / "protocol.json").write_text(
            json.dumps(
                dict(
                    epochs=100,
                    base_epochs=50,
                    manifest_sha256=hashlib.sha256(b"{}").hexdigest(),
                )
            )
        )

    def prepare_continuation(self):
        original = self.run / "source50"
        for base in (self.repository, original):
            (base / "models").mkdir(parents=True, exist_ok=True)
            (base / "dataset.py").write_text("same loader")
            (base / "models/network.py").write_text("same model")
        predecessors = {
            "scratch50-qutcc": dict(job_id="51", command=["original qutcc"]),
            "scratch50-im2im": dict(job_id="52", command=["original im2im"]),
        }
        (self.run / "jobs_50epochs.json").write_text(json.dumps(predecessors))
        (self.run / "jobs.json").write_text(json.dumps(predecessors))

    def test_fresh_plan_has_two_smoke_gates_and_complete_retry_submits_nothing(self):
        with patch.object(
            slurm.subprocess,
            "check_output",
            side_effect=["11;unicorn\n", "12\n", "13\n", "14\n"],
        ) as sbatch:
            jobs = submit.submit_training(self.run, self.project, self.source)
        self.assertEqual(len(jobs), 4)
        commands = [call.args[0] for call in sbatch.call_args_list]
        self.assertIn("--smoke", commands[0])
        self.assertIn("--time=00:20:00", commands[0])
        self.assertIn("--dependency=afterok:11", commands[1])
        self.assertIn("--dependency=afterok:13", commands[3])
        self.assertNotIn("--smoke", commands[1])
        self.assertEqual(json.loads((self.run / "jobs.json").read_text()), jobs)
        with patch.object(slurm.subprocess, "check_output") as retry:
            self.assertEqual(
                submit.submit_training(self.run, self.project, self.source), jobs
            )
        retry.assert_not_called()

    def test_partial_fresh_submission_reuses_accepted_smoke_job(self):
        failure = subprocess.CalledProcessError(
            1, "sbatch", output="temporary scheduler failure"
        )
        with patch.object(
            slurm.subprocess, "check_output", side_effect=["11\n", failure]
        ):
            with self.assertRaisesRegex(RuntimeError, "temporary scheduler failure"):
                submit.submit_training(self.run, self.project, self.source)
        saved = json.loads((self.run / "jobs.json").read_text())
        self.assertEqual(list(saved), ["scratch100-qutcc-smoke"])
        with patch.object(
            slurm.subprocess, "check_output", side_effect=["12\n", "13\n", "14\n"]
        ) as retry:
            submit.submit_training(self.run, self.project, self.source)
        self.assertEqual(retry.call_count, 3)
        self.assertIn("--dependency=afterok:11", retry.call_args_list[0].args[0])

    def test_missing_or_changed_manifest_fails_before_any_submission(self):
        with patch.object(slurm.subprocess, "check_output") as sbatch:
            with self.assertRaises(FileNotFoundError):
                submit.submit_training(self.run, self.root / "absent", self.source)
            self.manifest.write_text("changed")
            with self.assertRaisesRegex(ValueError, "manifest"):
                submit.submit_training(self.run, self.project, self.source)
        sbatch.assert_not_called()

    def test_changed_recorded_command_is_rejected_before_duplicate_submission(self):
        with patch.object(
            slurm.subprocess, "check_output", side_effect=["11", "12", "13", "14"]
        ):
            submit.submit_training(self.run, self.project, self.source)
        records = json.loads((self.run / "jobs.json").read_text())
        records["scratch100-qutcc-smoke"]["command"].append("unexpected-change")
        (self.run / "jobs.json").write_text(json.dumps(records))
        with patch.object(slurm.subprocess, "check_output") as sbatch:
            with self.assertRaisesRegex(ValueError, "Recorded submission differs"):
                submit.submit_training(self.run, self.project, self.source)
        sbatch.assert_not_called()

    def test_partial_continuation_reuses_jobs_and_keeps_predecessor_records(self):
        self.prepare_continuation()
        failure = subprocess.CalledProcessError(1, "sbatch", output="busy")
        with patch.object(
            slurm.subprocess, "check_output", side_effect=["101", failure]
        ):
            with self.assertRaises(RuntimeError):
                schedule.schedule_continuations(self.run, self.source)
        with patch.object(
            slurm.subprocess, "check_output", return_value="102"
        ) as retry:
            schedule.schedule_continuations(self.run, self.source)
        self.assertEqual(retry.call_count, 1)
        self.assertIn("--dependency=afterok:52", retry.call_args.args[0])
        records = json.loads((self.run / "jobs.json").read_text())
        self.assertEqual(
            set(records),
            {
                "scratch50-qutcc",
                "scratch50-im2im",
                "scratch100-qutcc",
                "scratch100-im2im",
            },
        )
        self.assertEqual(records["scratch50-qutcc"]["job_id"], "51")
        self.assertEqual(records["scratch100-qutcc"]["job_id"], "101")
        with patch.object(slurm.subprocess, "check_output") as complete_retry:
            schedule.schedule_continuations(self.run, self.source)
        complete_retry.assert_not_called()

    def test_continuation_checks_both_dependencies_before_first_submission(self):
        self.prepare_continuation()
        path = self.run / "jobs_50epochs.json"
        records = json.loads(path.read_text())
        records["scratch50-im2im"]["job_id"] = "invalid"
        path.write_text(json.dumps(records))
        with patch.object(slurm.subprocess, "check_output") as sbatch:
            with self.assertRaisesRegex(ValueError, "predecessor"):
                schedule.schedule_continuations(self.run, self.source)
        sbatch.assert_not_called()

    def test_added_removed_or_changed_training_code_is_rejected(self):
        self.prepare_continuation()
        original = self.run / "source50"
        for base in (self.repository, original):
            extra = base / "models/extra.py"
            extra.write_text("new file")
            with self.assertRaisesRegex(ValueError, "code changed"):
                slurm.require_same_training_code(self.repository, original)
            extra.unlink()
        (self.repository / "models/network.py").write_text("changed model")
        with patch.object(slurm.subprocess, "check_output") as sbatch:
            with self.assertRaisesRegex(ValueError, "code changed"):
                schedule.schedule_continuations(self.run, self.source)
        sbatch.assert_not_called()

    def test_submission_lock_prevents_concurrent_plans(self):
        with slurm.submission_lock(self.run):
            with self.assertRaises(BlockingIOError):
                with slurm.submission_lock(self.run):
                    self.fail("A concurrent submission acquired the same lock")


if __name__ == "__main__":
    unittest.main()
