"""Create synthetic tensors to exercise the training CLI; these are not dataset results."""

import argparse
from pathlib import Path

import torch

from .model import DProjection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.manual_seed(12)
    centers = torch.randn(12, 8)
    train = torch.randn(96, 8)
    raw = (train @ centers.T / centers.square().sum(-1)).softmax(-1)
    mean, scale = raw.mean(0), raw.std(0, unbiased=False)
    projection = DProjection(centers, mean, scale)
    weight, bias = torch.randn(3, 12), torch.randn(3)
    pack = {"centers": centers, "mean": mean, "scale": scale, "weight": weight, "bias": bias,
            "metadata": {"kind": "synthetic_smoke_only", "backbone": "random_features"}}
    for split, features in [("train", train), ("val", torch.randn(24, 8)), ("test", torch.randn(30, 8))]:
        pack[split] = {"features": features, "labels": (projection(features) @ weight.T + bias).argmax(-1)}
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(pack, path)


if __name__ == "__main__":
    main()
