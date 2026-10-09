import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from scripts import corrected_runtime as runtime


class RuntimeTests(unittest.TestCase):
    def test_gate_rejects_unverified_slow_or_different_hardware(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory)/'report.json'
            with patch.object(runtime, 'REPORT', report), patch('torch.cuda.get_device_name', return_value='test GPU'):
                self.assertEqual(runtime.select_runtime('mc_020')['version'], 'baseline')
                baseline = dict(arm='mc_010', fast=False, seconds_per_step=1., gpu='test GPU', torch_version=torch.__version__)
                faster = dict(baseline, fast=True, seconds_per_step=.5, prediction_mae_vs_original=.0001, resume_layout_verified=True, reference_checkpoint_epoch=10)
                for changed in [dict(resume_layout_verified=False), dict(seconds_per_step=1.1), dict(gpu='other GPU'), dict(prediction_mae_vs_original=.1)]:
                    report.write_text(json.dumps([baseline, dict(faster, **changed)]))
                    self.assertEqual(runtime.select_runtime('mc_020')['version'], 'baseline')
                report.write_text(json.dumps([baseline, faster]))
                self.assertTrue(runtime.select_runtime('mc_020')['fused_adamw'])
                self.assertEqual(runtime.select_runtime('conditioned')['version'], 'baseline')

    def test_resume_preserves_moments_and_step(self):
        model = torch.nn.Conv2d(4, 2, 3)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        model(torch.ones(2, 4, 8, 8)).square().mean().backward()
        optimizer.step()
        state = optimizer.state_dict()
        model.to(memory_format=torch.channels_last)
        target = torch.optim.AdamW(model.parameters(), lr=.5)
        runtime.restore_optimizer(target, state, {'fused_adamw': False})
        self.assertEqual(target.param_groups[0]['lr'], 1e-4)
        for parameter in model.parameters():
            torch.testing.assert_close(target.state[parameter]['exp_avg'], optimizer.state[parameter]['exp_avg'])
            self.assertEqual(float(target.state[parameter]['step']), 1.)
            self.assertEqual(target.state[parameter]['exp_avg'].stride(), parameter.stride())


if __name__ == '__main__':
    unittest.main()
