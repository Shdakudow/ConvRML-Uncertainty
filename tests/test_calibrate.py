"""Calibration reporting uses untouched test targets and raw reconstruction metrics."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset

from calibration.calibration_run import evaluate, compact_parameters

class CalibrationReportTests(unittest.TestCase):
    def test_raw_metrics_coverage_crossing_and_interval_intersection(self):
        class Model(nn.Module):
            pass
        class Calibration:
            model=Model()
            def predict(self,image):
                center=torch.full_like(image,1.5)
                lower=torch.full_like(image,.8)
                upper=torch.full_like(image,1.2)
                # A wholly out-of-range interval stays uncovered after intersection.
                lower[0]=1.1
                return center,lower,upper
        class Perceptual(nn.Module):
            def forward(self,prediction,target):
                self.maximum=float(prediction.max())
                return (prediction-target).abs().mean((1,2,3))
            def reset(self): pass
        images=torch.ones(2,3,33,35)
        perceptual=Perceptual()
        metrics=evaluate(Calibration(),DataLoader(TensorDataset(images,images),batch_size=2),perceptual,'cpu')
        self.assertAlmostEqual(metrics['coverage'],.5)
        self.assertAlmostEqual(metrics['mean_interval_width'],.25,places=6)
        self.assertAlmostEqual(metrics['mean_interval_width_in_image_range'],.1,places=6)
        self.assertAlmostEqual(metrics['psnr'],6.0206,places=4)
        self.assertLess(metrics['ssim'],1)
        self.assertEqual(perceptual.maximum,1)
        self.assertEqual(metrics['n_images'],2)
        summary=compact_parameters(dict(lambda_map=torch.tensor([[.1,.3]]),calibration_indices=[1,2,3],hb_pvalue=.05))
        self.assertEqual(summary['lambda_map']['shape'],[1,2])
        self.assertEqual(summary['calibration_indices_count'],3)
        json.dumps(summary)

    def test_entrypoint_keeps_calibration_and_test_splits_separate_and_reuses_fit(self):
        import calibration.calibration_run as calibrate
        class Calibration:
            calls=0
            def __init__(self,model): self.model=model;self.parameters={}
            def fit(self,loader,device):
                type(self).calls+=1
                self.parameters=dict(lower_q=.03,upper_q=.97,calibration_coverage=.9)
        dataset=TensorDataset(torch.zeros(10,3,4,4),torch.zeros(10,3,4,4))
        roles=[]
        def pairs(manifest,role):
            roles.append(role)
            return dataset
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            model=nn.Conv2d(3,3,1)
            source=root/'best.pt'
            torch.save(dict(model=model.state_dict(),method='qutcc',height=4,width=4,dataset_size=10,epoch=49,selected_through_epoch=50,validation_ssim=.8),source)
            manifest=root/'split.json'
            manifest.write_text('{}')
            args=['calibrate.py','--manifest',str(manifest),'--checkpoint',str(source),'--output',str(root/'out')]
            with patch('sys.argv',args),patch.object(calibrate,'make_model',return_value=model),patch.object(calibrate,'ImagePairs',side_effect=pairs),patch.object(calibrate,'QUTCCCalibration',Calibration),patch.object(calibrate,'LearnedPerceptualImagePatchSimilarity',return_value=nn.Identity()),patch.object(calibrate,'evaluate',return_value=dict(ssim=.8,coverage=.9)),patch('builtins.print'):
                calibrate.main()
                calibrate.main()
                # A different manifest must trigger a fresh fit.
                manifest.write_text('{"changed":true}')
                calibrate.main()
            self.assertEqual(Calibration.calls,2)
            self.assertEqual(roles,['calibration','test','test','calibration','test'])
            summary=json.loads((root/'out'/'summary.json').read_text())
            self.assertEqual(summary['selected_epoch'],49)
            self.assertEqual(summary['selected_through_epoch'],50)
            self.assertFalse(summary['training_weights_changed'])
            self.assertFalse(summary['test_used_for_fitting'])

if __name__=='__main__': unittest.main()
