"""Compare fixed-seed initialization, predictions, gradients and RNG across a refactor.

Capture before editing, then compare afterward. Random synthetic inputs only;
no dataset, GPU allocation, checkpoint write, or training job is required.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
CASES = ["im2im", "ensemble", "mc_dropout", "qutcc", "raw_im2im"]


def fingerprint(tensor):
    return hashlib.sha256(
        tensor.detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest()


def case(name):
    import torch

    torch.set_num_threads(2)
    torch.manual_seed(20261021)
    torch.use_deterministic_algorithms(True)
    paper = name in ("qutcc", "raw_im2im")
    if name == "qutcc":
        from experiments.parallel_lensless.models.qutcc import QUTCC

        model = QUTCC()
        # Nonidentity modulation ensures q conditioning is actually exercised.
        with torch.no_grad():
            for film in model.net.films.values():
                film.projection[-1].weight.normal_(0, 0.002)
                film.projection[-1].bias.normal_(0, 0.002)
    elif name == "raw_im2im":
        from experiments.raw_diffusercam.models.im2im import Im2im

        model = Im2im()
    elif name == "im2im":
        from experiments.parallel_lensless.models.im2im import Im2im

        model = Im2im()
    elif name == "mc_dropout":
        from experiments.parallel_lensless.models.mc_dropout import MCDropout

        model = MCDropout(0.05)
    else:
        from experiments.parallel_lensless.models.ensemble import EnsembleMember

        model = EnsembleMember()
    state = model.state_dict()
    initial = hashlib.sha256()
    schema = []
    for key, value in state.items():
        schema.append([key, list(value.shape), str(value.dtype)])
        initial.update(key.encode())
        initial.update(bytes.fromhex(fingerprint(value)))
    result = dict(
        state_schema_sha256=hashlib.sha256(json.dumps(schema).encode()).hexdigest(),
        initial_state_sha256=initial.hexdigest(),
        tensor_count=len(schema),
        predictions=[],
    )
    model.train()
    for shape in [(32, 32), (33, 47)] if paper else [(32, 32)]:
        model.zero_grad(set_to_none=True)
        y = torch.rand(1, 3, *shape, requires_grad=True)
        target = torch.rand_like(y)
        if name == "qutcc":
            outputs = tuple(model(y, y.new_full((1,), q)) for q in [0.05, 0.5, 0.95])
        elif name in ("im2im",):
            outputs = model.bounds(y)
        elif name == "raw_im2im":
            z = model(y)
            outputs = (z[:, 0], z[:, 1], z[:, 2])
        else:
            z = model(y)
            outputs = z if isinstance(z, tuple) else (z,)
        loss = sum((value - target).square().mean() for value in outputs)
        loss.backward()
        gradients = hashlib.sha256()
        count = 0
        for key, parameter in model.named_parameters():
            if parameter.grad is not None:
                gradients.update(key.encode())
                gradients.update(bytes.fromhex(fingerprint(parameter.grad)))
                count += 1
        result["predictions"].append(
            dict(
                shape=list(shape),
                output_sha256=[fingerprint(x) for x in outputs],
                loss=float(loss.detach()),
                input_gradient_sha256=fingerprint(y.grad),
                parameter_gradients_sha256=gradients.hexdigest(),
                gradient_count=count,
                rng_sha256=fingerprint(torch.get_rng_state()),
            )
        )
    result["torch"] = torch.__version__
    print(json.dumps(result))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--capture", type=Path)
    p.add_argument("--compare", type=Path)
    p.add_argument("--report", type=Path)
    p.add_argument("--case", choices=CASES)
    args = p.parse_args()
    if args.case:
        case(args.case)
        return
    if bool(args.capture) == bool(args.compare):
        p.error("Choose --capture or --compare")
    result = {}
    for name in CASES:
        output = subprocess.check_output(
            [sys.executable, str(Path(__file__)), "--case", name], cwd=ROOT
        )
        result[name] = json.loads(output)
        print("Checked", name, flush=True)
    if args.capture:
        args.capture.write_text(json.dumps(result, indent=2) + "\n")
    else:
        expected = json.loads(args.compare.read_text())
        assert (
            result == expected
        ), "Fixed-seed results changed; inspect case fingerprints before publication"
        if args.report:
            args.report.write_text(
                json.dumps(
                    dict(
                        status="pass",
                        comparison="exact CPU fingerprints before/after publication model interfaces",
                        seed=20261021,
                        device="cpu",
                        checks=[
                            "state dictionary keys/shapes",
                            "initialized weights",
                            "predictions",
                            "input gradients",
                            "parameter gradients",
                            "RNG consumption",
                        ],
                        limitations="Synthetic inputs on the same CPU/software environment; does not promise bitwise equality across GPU hardware or certify a full training run.",
                        cases=result,
                    ),
                    indent=2,
                )
                + "\n"
            )
        print(
            "PASS: initialization, predictions, losses, all parameter/input gradients and RNG match exactly."
        )


if __name__ == "__main__":
    main()
