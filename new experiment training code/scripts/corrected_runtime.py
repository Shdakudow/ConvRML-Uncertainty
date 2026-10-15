"""Benchmark-gated runtime changes, preserving the training objective and batch size."""
import json
from pathlib import Path
import torch

REPORT = Path('reports/runtime_benchmark_20261009.json')


def select_runtime(arm):
    result = dict(version='baseline', channels_last=False, fused_adamw=False)
    if not REPORT.exists():
        return result
    rows = json.loads(REPORT.read_text())
    representative = 'ensemble_00' if arm.startswith('ensemble_') else 'mc_010' if arm.startswith('mc_') else arm
    rows = [r for r in rows if r['arm'] == representative
            and r['gpu'] == torch.cuda.get_device_name()
            and r['torch_version'] == torch.__version__]
    original = next((r for r in rows if not r['fast']), None)
    fast = next((r for r in rows if r['fast']), None)
    if (original and fast and fast.get('resume_layout_verified') and fast.get('reference_checkpoint_epoch', 0) >= 10
            and fast['prediction_mae_vs_original'] <= .002
            and fast['seconds_per_step'] < .9 * original['seconds_per_step']):
        result = dict(version='channels_last_fused_v2', channels_last=True,
                      fused_adamw=True, benchmark=str(REPORT),
                      benchmark_speedup=original['seconds_per_step']/fast['seconds_per_step'])
    return result


def transfer(batch, runtime):
    y, x = batch[:, 0].cuda(non_blocking=True), batch[:, 1].cuda(non_blocking=True)
    if runtime.get('channels_last'):
        y = y.contiguous(memory_format=torch.channels_last)
    return y, x


def restore_optimizer(optimizer, state, runtime):
    # AdamW casts the scalar step state correctly when these flags are set BEFORE loading.
    # Overrides checkpoint flags so a job can move between benchmarked GPU types safely.
    for group in state['param_groups']:
        group['fused'] = runtime['fused_adamw']
        group['foreach'] = None
    optimizer.load_state_dict(state)
    # A saved NCHW moment keeps its old strides when loaded into an NHWC model.
    # Fused AdamW requires moments, parameters, and gradients to have matching layouts.
    for parameter, values in optimizer.state.items():
        for key in ('exp_avg', 'exp_avg_sq', 'max_exp_avg_sq'):
            value = values.get(key)
            if value is not None and value.stride() != parameter.stride():
                values[key] = torch.empty_like(parameter, memory_format=torch.preserve_format).copy_(value)
