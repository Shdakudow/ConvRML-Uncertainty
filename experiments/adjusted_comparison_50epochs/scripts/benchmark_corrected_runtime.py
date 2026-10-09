"""Isolated GPU benchmark; never reads or writes production checkpoints."""
import gc
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from scripts.corrected_comparison_study import load_model, loaders, training_loss
from scripts.corrected_runtime import restore_optimizer


def check_optimizer_resume():
    import copy
    model = torch.nn.Linear(4, 2).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=False)
    for fused in [False, True, False]:
        state = copy.deepcopy(optimizer.state_dict())
        for values in state['state'].values():
            for key, value in values.items():
                if isinstance(value, torch.Tensor):
                    values[key] = value.cpu()
        resumed = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=fused)
        restore_optimizer(resumed, state, {'fused_adamw': fused})
        resumed.zero_grad(set_to_none=True)
        loss = model(torch.ones(2, 4, device='cuda')).square().mean()
        loss.backward()
        resumed.step()
        if not all(torch.isfinite(p).all() for p in model.parameters()):
            raise RuntimeError('Optimizer resume produced nonfinite weights')
        optimizer = resumed
    if not all(float(s['step']) == 3 for s in optimizer.state.values()):
        raise RuntimeError('Optimizer resume did not preserve step counters')


def main():
    torch.set_num_threads(2)
    check_optimizer_resume()
    dataset, _ = loaders('rml')
    batch = torch.stack([dataset[0], dataset[1]]).pin_memory()
    results = []
    for arm in ['ensemble_00', 'conditioned', 'im2im', 'mc_010']:
        reference = None
        for fast in [False, True]:
            torch.backends.cudnn.benchmark = fast
            model, _ = load_model('rml', arm, epoch=10)
            model.cuda()
            if fast:
                model.to(memory_format=torch.channels_last)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4, fused=fast)
            y, x = batch[:, 0].cuda(), batch[:, 1].cuda()
            if fast:
                y = y.contiguous(memory_format=torch.channels_last)
            model.eval()
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                prediction = model.bounds(y)[0] if arm == 'im2im' else model(y)
                if arm.startswith('ensemble_'):
                    prediction = prediction[0]
                prediction = prediction.float().cpu()
            if reference is None:
                reference = prediction
            delta = float((prediction - reference).abs().mean())
            if not torch.isfinite(prediction).all() or delta > 0.01:
                raise RuntimeError(f'{arm}: excessive layout difference {delta}')
            model.train()
            times = []
            torch.cuda.reset_peak_memory_stats()
            for i in range(25):
                torch.cuda.synchronize()
                begin = time.perf_counter()
                y = batch[:, 0].cuda(non_blocking=fast)
                x = batch[:, 1].cuda(non_blocking=fast)
                if fast:
                    y = y.contiguous(memory_format=torch.channels_last)
                opt.zero_grad(set_to_none=True)
                q = torch.full((len(y),), 0.37, device='cuda') if arm == 'conditioned' else None
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    loss = training_loss(model, arm, y, x, q)
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite benchmark loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                opt.step()
                torch.cuda.synchronize()
                if i >= 5:
                    times.append(time.perf_counter() - begin)
            row = dict(arm=arm, fast=fast, seconds_per_step=sum(times)/len(times),
                       prediction_mae_vs_original=delta, final_loss=float(loss),
                       peak_gib=torch.cuda.max_memory_allocated()/2**30,
                       gpu=torch.cuda.get_device_name(), torch_version=torch.__version__, resume_verified=True,
                       reference_checkpoint_epoch=10)
            results.append(row)
            print(json.dumps(row), flush=True)
            del model, opt, loss, prediction, y, x
            gc.collect()
            torch.cuda.empty_cache()
    out = Path('reports/runtime_benchmark_20261009.json')
    out.parent.mkdir(exist_ok=True)
    temporary = out.with_suffix('.tmp')
    temporary.write_text(json.dumps(results, indent=2))
    temporary.replace(out)


if __name__ == '__main__':
    main()
