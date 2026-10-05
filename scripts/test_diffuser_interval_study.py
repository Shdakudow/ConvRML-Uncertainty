import unittest
from unittest.mock import patch
from scripts import diffuser_interval_study as diffuser
from scripts.prepare_diffuser_intervals import source_paths


class DiffuserStudyTest(unittest.TestCase):
    def test_correct_measurement_and_aligned_target(self):
        measurement,target=source_paths(123)
        self.assertEqual(measurement.parent.name,'diffuser')
        self.assertEqual(measurement.name,'4x_img_123_cam_0.tiff')
        self.assertEqual(target.parent.name,'gt2dc')
        self.assertEqual(target.name,'warped_4x_undistorted_img_123_cam_2.tiff')

    def test_camera_routes_training_and_calibration_separately(self):
        study=diffuser.study
        original=(study.ROOT,study.DATA,study.CALIBRATION)
        try:
            with patch.object(study,'main') as main:
                diffuser.main()
                main.assert_called_once()
                self.assertEqual(study.ROOT,diffuser.ROOT)
                self.assertEqual(study.CALIBRATION,diffuser.CALIBRATION)
                args=study.command(7,2)
                self.assertIn(str(diffuser.DATA),args)
                self.assertIn(str(diffuser.ROOT/'quantiles'),args)
                self.assertNotIn(str(original[1]),args)
        finally:
            study.ROOT,study.DATA,study.CALIBRATION=original


if __name__=='__main__':unittest.main()
