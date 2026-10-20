"""Joint, from-scratch adaptive-q training with a 90% marginal coverage target.

Minimize width plus pinball fidelity, using a nonnegative adaptive multiplier
for observed RGB-pixel miscoverage. A differentiable miss proxy supplies primal
gradients; hard empirical miss rate controls the multiplier. This is a training
heuristic for a constrained objective, not a statistical coverage guarantee.
"""

import torch
from torch import nn
from models.image_adaptive_qutcc import ImageAdaptiveQUTCC
from models.formal_quantile import pinball

PROTOCOL = "joint_adaptive_q_coverage90_scratch50_v1"


class CoverageController(nn.Module):
    def __init__(self, alpha=0.1, rate=0.005, momentum=0.99):
        super().__init__()
        self.alpha, self.rate, self.momentum = alpha, rate, momentum
        self.register_buffer("multiplier", torch.tensor(1.0))
        self.register_buffer("ema_risk", torch.tensor(alpha))
        self.register_buffer("updates", torch.tensor(0, dtype=torch.long))

    @torch.no_grad()
    def update(self, hard_risk):
        risk = hard_risk.detach().float()
        if not torch.isfinite(risk) or not 0 <= float(risk) <= 1:
            raise ValueError("Invalid risk")
        if self.updates == 0:
            self.ema_risk.copy_(risk)
        else:
            self.ema_risk.mul_(self.momentum).add_(risk, alpha=1 - self.momentum)
        self.multiplier.add_(self.rate * (self.ema_risk - self.alpha)).clamp_(min=0)
        self.updates.add_(1)


def interval_terms(lower, upper, target, tau=1 / 255):
    if tau <= 0:
        raise ValueError("tau must be positive")
    # Zero exactly on covered targets (including background pixels at 0 or 1).
    # A sigmoid miss proxy incorrectly assigns 50% miss probability at such ties.
    gap = torch.relu(lower - target) + torch.relu(target - upper)
    return dict(
        width=(upper - lower).mean(),
        soft_risk=(gap / (gap + tau)).mean(),
        hard_risk=((target < lower) | (target > upper)).float().mean(),
    )


class ConstrainedAdaptiveQUTCC(ImageAdaptiveQUTCC):
    def __init__(self, quantile_model):
        super().__init__(quantile_model, alpha=0.1)
        self.controller = CoverageController()
        self.width_weight = 0.1
        self.tau = 1 / 255


def joint_step(model, y, x, optimizer, generator=None, amp=True):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    q = model.training_quantiles(y, generator)
    with torch.autocast(y.device.type, dtype=torch.bfloat16, enabled=amp):
        reconstruction_loss = pinball(model(y, q).float(), x, q)
    if not torch.isfinite(reconstruction_loss):
        raise ValueError("Nonfinite pinball")
    # Accumulate gradients before one optimizer update; splitting backwards
    # releases the random-q graph before building the two endpoint graphs.
    reconstruction_loss.backward()
    with torch.autocast(y.device.type, dtype=torch.bfloat16, enabled=amp):
        outputs = model.interval_prediction(y)
        terms = interval_terms(outputs["lower"], outputs["upper"], x, model.tau)
        multiplier = model.controller.multiplier.detach().clone()
        interval_objective = model.width_weight * (
            terms["width"] + multiplier * terms["soft_risk"]
        )
    if not torch.isfinite(interval_objective):
        raise ValueError("Nonfinite interval objective")
    # Both backbone and policy receive the width/coverage gradient.
    interval_objective.backward()
    selector_grad = torch.linalg.vector_norm(
        torch.stack(
            [
                p.grad.float().norm()
                for p in model.selector.parameters()
                if p.grad is not None
            ]
        )
    )
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    model.controller.update(terms["hard_risk"])
    return dict(
        pinball=float(reconstruction_loss.detach()),
        width=float(terms["width"].detach()),
        soft_risk=float(terms["soft_risk"].detach()),
        hard_risk=float(terms["hard_risk"]),
        multiplier=float(model.controller.multiplier),
        risk_ema=float(model.controller.ema_risk),
        selector_grad=float(selector_grad),
        q_lower=float(outputs["q_lower"].detach().mean()),
        q_upper=float(outputs["q_upper"].detach().mean()),
    )
