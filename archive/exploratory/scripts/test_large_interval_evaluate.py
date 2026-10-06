"""CPU synthetic integration check for calibration/report plumbing."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from scripts import large_interval_evaluate as ev


class EvaluationTest(unittest.TestCase):
    def test_calibration_uses_calibration_targets_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);data=root/'data';cal=root/'cal'
            data.mkdir();cal.mkdir()
            (data/'manifest.json').write_text(json.dumps({'train':[1],'val':[2],'test':[3]}))
            (cal/'manifest.json').write_text(json.dumps({'calibration':[4,5]}))
            pairs=np.full((3,2,3,256,256),100,dtype=np.uint8)
            pairs[-1,1]=151  # Test residual differs from calibration residual.
            np.save(data/'pairs.npy',pairs)
            pairs=np.full((2,2,3,256,256),100,dtype=np.uint8)
            pairs[:,1]=125
            np.save(cal/'pairs.npy',pairs)
            def predict(y):return y,y-.1,y+.1
            with patch.object(ev,'ROOT',root/'results'),patch.object(ev,'DATA',data),patch.object(ev,'CALIBRATION',cal),\
                 patch.object(ev,'load_predictor',return_value=(predict,1.)),\
                 patch.object(torch.Tensor,'cuda',lambda self,*a,**kw:self),\
                 patch.object(torch,'autocast',lambda *a,**kw:contextlib.nullcontext()),\
                 patch.object(torch.cuda,'get_device_name',return_value='CPU test'),\
                 contextlib.redirect_stdout(io.StringIO()):
                ev.evaluate(0,5)
            report=json.loads((root/'results/evaluation/epoch_005/ensemble.json').read_text())
            self.assertAlmostEqual(report['factor'],(25/255)/.1,places=5)
            self.assertAlmostEqual(report['metrics']['width'],50/255,places=5)
            self.assertEqual(report['metrics']['coverage'],0.)
            self.assertEqual(report['test_count'],1)
            self.assertFalse(list(root.rglob('*scores*')))


if __name__=='__main__':unittest.main()
