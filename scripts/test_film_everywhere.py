"""Check identity initialization, gradients and reconstruction isolation."""
import copy
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch import nn
from models.formal_quantile import QuantileRML, pinball
from models.film_everywhere import EverywhereQuantileRML

class ConvNeXtBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_dw = nn.Conv2d(256, 256, 1, groups=256)
    def forward(self, x):
        return x + self.conv_dw(x)

class SmallBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(3, 256, 1), ConvNeXtBlock())
        self.refine = nn.Sequential(nn.Conv2d(256, 256, 1), nn.GELU())
        self.up2 = nn.Identity()
        self.head = nn.Conv2d(256, 3, 3, padding=1)
    def features(self, y):
        return self.refine(self.encoder(y))

class Tests(unittest.TestCase):
    def test_identity_gradients_and_isolation(self):
        torch.manual_seed(42)
        old = QuantileRML(SmallBase(), 'conditioned', levels=(.05, .95)).eval()
        new = EverywhereQuantileRML(copy.deepcopy(old.base)).eval()
        new.load_original(old.state_dict())
        y = torch.rand(2, 3, 8, 8)
        for before, after in zip(old.bounds(y), new.bounds(y)):
            torch.testing.assert_close(before, after, rtol=0, atol=0)
        q = torch.tensor([.05, .95])
        pinball(new.predict(y, q=q), torch.rand_like(y), q).backward()
        for film in new.deep_films.values():
            self.assertGreater(film.affine.weight.grad.abs().sum().item(), 0)
        r = new.features(y)[1].detach()
        with torch.no_grad():
            for film in new.deep_films.values():
                film.affine.bias.add_(.2)
        changed = new.predict(y, q=q)
        self.assertFalse(torch.allclose(changed, old.predict(y, old.features(y)[0], q)))
        torch.testing.assert_close(new.features(y)[1], r, rtol=0, atol=0)
        self.assertIsNone(new._embedding)
        restored = EverywhereQuantileRML(SmallBase()).eval()
        restored.load_state_dict(new.state_dict(), strict=True)
        torch.testing.assert_close(restored.predict(y, q=q), changed)

    def test_context_cleared_on_exception(self):
        model = EverywhereQuantileRML(SmallBase())
        with self.assertRaises(RuntimeError):
            with model.condition(torch.tensor([.5])):
                raise RuntimeError('simulated failure')
        self.assertIsNone(model._embedding)

    def test_sequential_backward_matches_original_loss(self):
        torch.manual_seed(12)
        old = QuantileRML(SmallBase(), 'conditioned', levels=(.05, .95))
        new = EverywhereQuantileRML(copy.deepcopy(old.base))
        new.load_original(old.state_dict())
        y, x = torch.rand(2, 3, 8, 8), torch.rand(2, 3, 8, 8)
        q = torch.tensor([.2, .8])
        f, r = old.features(y)
        ((r-x).square().mean() + pinball(old.predict(y, f, q), x, q)).backward()
        (new.features(y)[1]-x).square().mean().backward()
        pinball(new.predict(y, q=q), x, q).backward()
        new_parameters = dict(new.named_parameters())
        for name, parameter in old.named_parameters():
            torch.testing.assert_close(parameter.grad, new_parameters[name].grad,
                                       rtol=1e-4, atol=1e-7)

if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
