"""Ensure the exact reviewed input and unchanged target are loaded every epoch."""
import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from PIL import Image
import data
class FixedDataset(unittest.TestCase):
    def test_reads_materialized_pixels_and_target_exactly(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);measurement=np.full((40,60,3),73,dtype=np.uint8);target=np.full_like(measurement,199)
            Image.fromarray(measurement).save(root/'m.png');Image.fromarray(target).save(root/'x.png')
            row={'id':'sample','measurement':{'path':str(root/'m.png')},'target':{'path':str(root/'x.png')}}
            path=root/'manifest.json';path.write_text(json.dumps({'roles':{'train':[row]}}))
            with patch.object(data,'MANIFEST',path):
                pairs=data.Pairs('raw','train');first=pairs[0];pairs.set_epoch(9);last=pairs[0]
            self.assertTrue(torch.equal(first,last))
            self.assertTrue(torch.equal((first[0]*255).round().byte().permute(1,2,0),torch.from_numpy(measurement)))
            self.assertTrue(torch.equal((first[1]*255).round().byte().permute(1,2,0),torch.from_numpy(target)))
if __name__=='__main__':unittest.main()
