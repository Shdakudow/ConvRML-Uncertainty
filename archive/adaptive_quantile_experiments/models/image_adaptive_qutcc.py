"""Image-adaptive q, trained alongside a continuous quantile reconstruction model.

Two updates per batch: pinball updates QUTCC at detached training q values;
fixed-alpha interval score updates the image-dependent q policy through QUTCC.
This avoids minimizing pinball by changing its quantile label. No MSE term.
"""
from contextlib import contextmanager
import torch
from torch import nn
from models.learned_quantile_selector import ImageQuantileSelector, interval_score
from models.formal_quantile import pinball

PROTOCOL = 'image_adaptive_qutcc_alternating_pinball_interval_fp32q_v1'


class ImageAdaptiveQUTCC(nn.Module):
    def __init__(self, quantile_model, alpha=.1):
        super().__init__()
        if not 0 < alpha < 1:
            raise ValueError('alpha must lie in (0,1)')
        if getattr(quantile_model, 'q_selector', None) is not None:
            raise ValueError('Expected original QUTCC without a separate frozen selector')
        self.quantile_model = quantile_model.requires_grad_(True)
        self.selector = ImageQuantileSelector()
        self.alpha = alpha

    def forward(self, y, q=None):
        return self.quantile_model(y, q)

    def interval_prediction(self, y):
        # Keep small image-dependent q differences out of BF16 rounding.
        with torch.autocast(y.device.type, enabled=False):
            q_lower, q_upper = self.selector(y.float())
        with torch.no_grad():
            median = self(y).float().clamp(0, 1)
        lower = torch.minimum(self(y, q_lower).float(), median).clamp(0, 1)
        upper = torch.maximum(self(y, q_upper).float(), median).clamp(0, 1)
        return dict(prediction=median, lower=lower, upper=upper,
                    q_lower=q_lower, q_upper=q_upper)

    def bounds(self, y):
        result = self.interval_prediction(y)
        return result['prediction'], result['lower'], result['upper']

    @torch.no_grad()
    def training_quantiles(self, y, generator=None):
        lo, hi = self.selector(y)
        uniform = torch.rand(len(y), device=y.device, generator=generator)
        choice = torch.rand(len(y), device=y.device, generator=generator)
        # Half the queries cover the full distribution; a quarter target each
        # learned endpoint. q is detached for the backbone's pinball objective.
        return torch.where(choice < .5, uniform, torch.where(choice < .75, lo, hi)).detach()


@contextmanager
def frozen_for_policy(model):
    """Freeze weights temporarily, retaining d Q(y,q)/d q for the policy update."""
    base = model.quantile_model
    parameters = list(base.parameters())
    flags = [p.requires_grad for p in parameters]
    modes = [(m, m.training) for m in base.modules()]
    base.requires_grad_(False).eval()
    try:
        yield
    finally:
        for p, flag in zip(parameters, flags):
            p.requires_grad_(flag)
        for m, training in modes:
            m.training = training


def adaptive_step(model, y, x, backbone_optimizer, selector_optimizer, generator=None, amp=True):
    model.train()
    q = model.training_quantiles(y, generator)
    backbone_optimizer.zero_grad(set_to_none=True)
    selector_optimizer.zero_grad(set_to_none=True)
    with torch.autocast(y.device.type, dtype=torch.bfloat16, enabled=amp):
        quantile_loss = pinball(model(y, q).float(), x, q)
    if not torch.isfinite(quantile_loss):
        raise ValueError('Nonfinite pinball loss')
    quantile_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.quantile_model.parameters(), 1., error_if_nonfinite=True)
    backbone_optimizer.step()
    backbone_optimizer.zero_grad(set_to_none=True)
    with frozen_for_policy(model):
        with torch.autocast(y.device.type, dtype=torch.bfloat16, enabled=amp):
            result = model.interval_prediction(y)
            policy_loss = interval_score(result['lower'], result['upper'], x, model.alpha)
        if not torch.isfinite(policy_loss):
            raise ValueError('Nonfinite interval loss')
        policy_loss.backward()
        policy_grad = torch.nn.utils.clip_grad_norm_(model.selector.parameters(), 1., error_if_nonfinite=True)
        selector_optimizer.step()
    return dict(pinball=float(quantile_loss.detach()), interval_score=float(policy_loss.detach()),
                selector_grad_norm=float(policy_grad), q_lower=float(result['q_lower'].detach().mean()),
                q_upper=float(result['q_upper'].detach().mean()))
