"""Data integrity tests: deterministic pairings, unchanged targets and held-out families."""
import unittest
import torch
from corruptions import corrupt,seed_for,ALL_CONDITIONS,TRAIN_CONDITIONS,OOD_CONDITIONS
class Integrity(unittest.TestCase):
    def setUp(self):
        self.y=torch.linspace(0,1,3*70*110).reshape(3,70,110)
    def test_deterministic_shape_range_and_no_mutation(self):
        before=self.y.clone()
        for condition in ALL_CONDITIONS+('mixed',):
            for level in ('mild','moderate','severe'):
                a=corrupt(self.y,condition,level,42);b=corrupt(self.y,condition,level,42)
                self.assertTrue(torch.equal(a,b));self.assertEqual(a.shape,self.y.shape)
                self.assertTrue(torch.isfinite(a).all());self.assertGreaterEqual(float(a.min()),0);self.assertLessEqual(float(a.max()),1)
        self.assertTrue(torch.equal(self.y,before))
    def test_pair_target_unchanged(self):
        x=self.y.flip(-1).clone();pair=torch.stack((self.y,x));before=pair.clone()
        transformed=torch.stack((corrupt(pair[0],'mixed',seed=15),pair[1]))
        self.assertTrue(torch.equal(pair,before));self.assertTrue(torch.equal(transformed[1],x))
    def test_randomness_and_holdout(self):
        self.assertFalse(torch.equal(corrupt(self.y,'gaussian',seed=1),corrupt(self.y,'gaussian',seed=2)))
        self.assertFalse(set(TRAIN_CONDITIONS)&set(OOD_CONDITIONS))
        self.assertNotEqual(seed_for('train',1,1),seed_for('train',1,2))
        seen=set(corrupt(self.y,'mixed',seed=i,return_condition=True)[1] for i in range(100))
        self.assertFalse(seen&set(OOD_CONDITIONS));self.assertEqual(seen,set(TRAIN_CONDITIONS)|{'clean'})
    def test_clean_exact(self):self.assertTrue(torch.equal(self.y,corrupt(self.y,'clean')))
if __name__=='__main__':unittest.main()
