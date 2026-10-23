"""Exercise report identity, ensemble membership, safe cleanup, and reconnects."""

import copy
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import coverage90_cluster as cluster


def report(method="qutcc", epoch=5):
    members = 5 if method == "ensemble5" else 1
    return dict(
        request=dict(
            camera="rml",
            method=method,
            epoch=epoch,
            calibration="tight_empirical",
            test_used_for_calibration=False,
            endpoint_clipping=[0, 1],
            target_coverage=0.9,
        ),
        calibration=dict(
            n_images=2048,
            calibrated_on="calibration split only",
            confidence_guarantee=False,
            target_coverage=0.9,
            endpoint_clipping=[0, 1],
        ),
        test_count=5000,
        per_image=[dict(index=i, id=str(i)) for i in range(5000)],
        example_count=0,
        metrics={
            mode: dict(
                psnr=30.0, ssim=0.9, mae=0.02, coverage=0.9, width=0.1, crossing=0.0
            )
            for mode in ("raw", "clipped")
        },
        provenance=[
            dict(
                epoch=epoch,
                arm=f"ensemble_{i:02d}",
                seeds=dict(initialization=i, data=100 + i),
            )
            for i in range(members)
        ],
    )


class ClusterOperations(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name) / "run"
        self.root.mkdir()
        self.patch = patch.object(cluster, "RUN_ROOT", self.root)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.task = dict(index=1, group="rml", camera="rml", method="qutcc", epoch=5)

    def test_wrong_epoch_or_coverage_cannot_reuse_a_report(self):
        cluster.validate_report(self.task, report())
        with self.assertRaises(ValueError):
            cluster.validate_report(self.task, report(epoch=10))
        wrong = report()
        wrong["request"]["target_coverage"] = 0.95
        with self.assertRaises(ValueError):
            cluster.validate_report(self.task, wrong)

    def test_duplicate_test_images_and_missing_exports_are_rejected(self):
        wrong = report()
        wrong["per_image"][1]["id"] = "0"
        with self.assertRaises(ValueError):
            cluster.validate_report(self.task, wrong)
        with self.assertRaisesRegex(ValueError, "requested examples"):
            cluster.validate_report(self.task, report(), required_examples=50)

    def test_aggregate_metrics_must_use_the_full_saved_test_set(self):
        saved = report()
        for row in saved["per_image"]:
            row.update(copy.deepcopy(saved["metrics"]))
        cluster.validate_report(self.task, saved, check_aggregates=True)
        saved["metrics"]["clipped"]["coverage"] = 0.95
        with self.assertRaisesRegex(ValueError, "Aggregate metrics"):
            cluster.validate_report(self.task, saved, check_aggregates=True)

    def test_ensemble_requires_five_distinct_members_and_shuffle_seeds(self):
        task = dict(self.task, method="ensemble5")
        valid = report("ensemble5")
        cluster.validate_report(task, valid)
        wrong = copy.deepcopy(valid)
        wrong["provenance"][1]["arm"] = "ensemble_00"
        with self.assertRaisesRegex(ValueError, "duplicated ensemble"):
            cluster.validate_report(task, wrong)
        valid["provenance"][1]["seeds"]["data"] = 100
        with self.assertRaisesRegex(ValueError, "data-order"):
            cluster.validate_report(task, valid)

    def test_duplicate_archive_entry_does_not_hide_missing_member(self):
        entries = [dict(path=path) for path in sorted(cluster.checkpoint_paths(5))]
        cluster.validate_manifest(5, entries)
        entries[-1] = entries[0]
        with self.assertRaises(ValueError):
            cluster.validate_manifest(5, entries)

    def test_reconnect_after_lost_submission_response_does_not_submit_twice(self):
        active = []

        def sbatch(command, **kwargs):
            name = next(
                arg.split("=", 1)[1] for arg in command if arg.startswith("--job-name=")
            )
            active.append(dict(job="1234", name=name, state="RUNNING"))
            return "1234\n"

        request = dict(epoch=5, kind="baseline", tasks=[self.task])
        with (
            patch.object(cluster, "jobs", side_effect=lambda: list(active)),
            patch.object(cluster, "finished", return_value=False),
            patch.object(
                cluster.subprocess, "check_output", side_effect=sbatch
            ) as submit,
        ):
            with patch.object(
                cluster, "write_json", side_effect=OSError("response lost")
            ):
                with self.assertRaises(OSError):
                    cluster.submit(request)
            recovered = cluster.submit(request)
            self.assertEqual(recovered["job"], "1234")
            self.assertTrue(recovered["recovered"])
            cluster.submit(request)
            self.assertEqual(submit.call_count, 1)

    def test_finished_tasks_do_not_reserve_a_gpu(self):
        with (
            patch.object(cluster, "jobs", return_value=[]),
            patch.object(cluster, "finished", return_value=True),
            patch.object(cluster.subprocess, "check_output") as submit,
        ):
            result = cluster.submit(dict(epoch=5, kind="baseline", tasks=[self.task]))
            self.assertIsNone(result["job"])
            submit.assert_not_called()

    def test_another_epoch_cannot_exceed_the_stream_gpu_limit(self):
        active = [dict(job="555", name="cov90-baseline-010-other", state="RUNNING")]
        with (
            patch.object(cluster, "jobs", return_value=active),
            patch.object(cluster, "finished", return_value=False),
            patch.object(cluster.subprocess, "check_output") as submit,
        ):
            with self.assertRaisesRegex(RuntimeError, "still active"):
                cluster.submit(dict(epoch=5, kind="baseline", tasks=[self.task]))
            submit.assert_not_called()

    def test_cleanup_preserves_original_weights_and_waits_for_all_six_tasks(self):
        shared = self.root.parent / "experiment2_uncertainty_analysis_20261019/weights"
        entries = []
        for name in sorted(cluster.checkpoint_paths(5)):
            source = shared / name
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_bytes(b"checkpoint")
            entries.append(
                dict(
                    path=name, size=10, sha256=hashlib.sha256(b"checkpoint").hexdigest()
                )
            )
        self.assertEqual(
            cluster.inventory(dict(epoch=5, entries=entries))["missing"], []
        )
        originals = [shared / entry["path"] for entry in entries]
        copies = [self.root / "weights/epoch_005" / entry["path"] for entry in entries]
        self.assertEqual(originals[0].stat().st_ino, copies[0].stat().st_ino)
        tasks = [
            dict(group=camera, method=method, epoch=5)
            for camera in cluster.CAMERAS
            for method in ("im2im", "ensemble5", "mc_005")
        ]
        with patch.object(cluster, "finished", return_value=True):
            with self.assertRaises(ValueError):
                cluster.cleanup(dict(epoch=5, entries=entries, tasks=tasks[:1]))
            self.assertTrue(all(path.exists() for path in copies))
        with patch.object(cluster, "finished", return_value=False):
            with self.assertRaises(ValueError):
                cluster.cleanup(dict(epoch=5, entries=entries, tasks=tasks))
        with patch.object(cluster, "finished", return_value=True):
            cluster.cleanup(dict(epoch=5, entries=entries, tasks=tasks))
        self.assertTrue(all(path.read_bytes() == b"checkpoint" for path in originals))
        self.assertTrue(all(not path.exists() for path in copies))


if __name__ == "__main__":
    unittest.main()
