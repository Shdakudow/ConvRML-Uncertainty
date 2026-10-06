"""CPU/GPU-independent structural check for the ConvRML reconstruction model."""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.convnext import ConvRecon


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--height", type=int, default=300)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()

    model = ConvRecon(n_channels=3, height=args.height, width=args.width)
    model.train()
    x = torch.rand(args.batch_size, 3, args.height, args.width)
    y = model(x)
    loss = y.square().mean()
    loss.backward()

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    assert y.shape == x.shape, f"Expected {x.shape}, got {y.shape}"
    assert any(parameter.grad is not None for parameter in model.parameters())
    print(f"input={tuple(x.shape)} output={tuple(y.shape)} parameters={parameter_count:,}")


if __name__ == "__main__":
    main()
