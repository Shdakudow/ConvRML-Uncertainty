import sys, unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from interval_calibration import fit,apply,metrics

class CalibrationTests(unittest.TestCase):
    def test_biased_heteroscedastic(self):
        rng=np.random.default_rng(19)
        r=np.full((200,3,8,8),.5)
        w=rng.uniform(.01,.08,r.shape)
        x=r+w*rng.normal(.8,1,r.shape)
        lo,hi=r-w,r+w
        p=fit(x,r,lo,hi)
        self.assertGreater(p['upper'],p['lower'])
        for method in ('additive','shared','asymmetric'):
            l,h=apply(r,lo,hi,p,method)
            self.assertGreaterEqual(metrics(x,l,h)['coverage'],.9)
        l,h=apply(r,lo,hi,p,'asymmetric')
        self.assertLessEqual(metrics(x,l,h)['lower_tail'],.05)
        self.assertLessEqual(metrics(x,l,h)['upper_tail'],.05)
        np.testing.assert_array_equal(apply(r,lo,hi,p,'raw')[0],lo)

    def test_collapsed_crossed_and_invalid(self):
        r=np.zeros((4,3,2,2));x=np.ones_like(r)
        p=fit(x,r,r+1,r-1)
        l,h=apply(r,r+1,r-1,p,'asymmetric')
        self.assertTrue(np.isfinite(h).all())
        self.assertTrue((l<=h).all())
        self.assertEqual(metrics(x,l,h)['coverage'],1.)
        with self.assertRaises(ValueError):fit(x*float('nan'),r,r,r)

if __name__=='__main__':unittest.main()
