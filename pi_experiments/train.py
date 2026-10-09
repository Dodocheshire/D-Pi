"""Train Pi noise from an exported official DCBM using train/validation only."""

import argparse
import copy
import json
import time
import hashlib
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from .model import DPi, DProjection


def batches(split, batch_size, shuffle=False):
    return DataLoader(TensorDataset(split["features"].float(), split["labels"].long()),
                      batch_size=batch_size, shuffle=shuffle)


@torch.no_grad()
def evaluate(model, split, batch_size, samples, seed):
    model.eval()
    device = model.feature_scale.device
    correct = count = 0
    nll = drift = 0.0
    # Fixed MC draws make validation model comparisons reproducible.
    with torch.random.fork_rng(devices=[device.index or 0] if device.type == "cuda" else []):
        torch.manual_seed(seed)
        for features, labels in batches(split, batch_size):
            features, labels = features.to(device), labels.to(device)
            probabilities = []
            for _ in range(samples):
                logits, concepts, clean, _ = model(features, sample=True)
                probabilities.append(logits.softmax(-1))
                drift += (concepts - clean).square().mean().item() * len(labels) / samples
            probs = torch.stack(probabilities).mean(0)
            nll += F.nll_loss(probs.clamp_min(1e-12).log(), labels, reduction="sum").item()
            correct += (probs.argmax(-1) == labels).sum().item()
            count += len(labels)
    return {"accuracy": correct / count, "nll": nll / count,
            "standardized_concept_mse": drift / count,
            "nec_threshold_1e-5": (model.head.weight.abs() > 1e-5).float().sum(-1).mean().item(),
            "mc_samples": samples}


def select_state(model, loader, val, epochs, lr, args, head=False):
    params = model.head.parameters() if head else model.generator.parameters()
    optimizer = torch.optim.Adam([p for p in params if p.requires_grad], lr=lr)
    best_loss, best, history = float("inf"), copy.deepcopy(model.state_dict()), []
    started = time.monotonic()
    for epoch in range(epochs):
        model.train()
        for features, labels in loader:
            features, labels = features.to(args.device), labels.to(args.device)
            optimizer.zero_grad()
            if head:
                logits = model.head(features)
                loss = F.cross_entropy(logits, labels) + args.l1 * model.head.weight.abs().sum()
            else:
                logits, concepts, clean, energy = model(features)
                loss = F.cross_entropy(logits, labels)
                loss = loss + args.concept_weight * F.mse_loss(concepts, clean) + args.energy_weight * energy
            loss.backward()
            optimizer.step()
        metrics = evaluate(model, val, args.batch_size, args.mc_samples, args.seed + 1000)
        history.append({"epoch": epoch + 1, **metrics})
        if metrics["nll"] < best_loss:
            best_loss, best = metrics["nll"], copy.deepcopy(model.state_dict())
        print(f"{'head' if head else 'noise'} epoch={epoch+1}/{epochs} "
              f"val_acc={metrics['accuracy']:.5f} val_nll={metrics['nll']:.5f} "
              f"elapsed={time.monotonic()-started:.1f}s", flush=True)
    model.load_state_dict(best)
    return history


def run(pack, args):
    torch.set_num_threads(8)
    torch.manual_seed(args.seed)
    features = pack["train"]["features"].float()
    feature_scale = features.square().mean(0).sqrt().clamp_min(1e-6)
    projection = DProjection(pack["centers"], pack["mean"], pack["scale"])
    model = DPi(projection, pack["weight"], pack["bias"], feature_scale,
                args.mode, args.conditioning, args.site, args.hidden, args.initial_scale).to(args.device)
    baseline = DPi(copy.deepcopy(projection), pack["weight"], pack["bias"],
                   feature_scale, mode="none").to(args.device)
    history = {}
    if model.generator is not None:
        history["noise_adaptation"] = select_state(model, batches(pack["train"], args.batch_size, True),
                                                 pack["val"], args.epochs, args.lr, args)
    if args.refit_epochs:
        if model.generator is not None:
            model.generator.requires_grad_(False)
        # One fixed sampled concept vector per train image, matching the LF refit stage.
        with torch.no_grad():
            cached = []
            for x, _ in batches(pack["train"], args.batch_size):
                cached.append(model(x.to(args.device))[1].cpu())
        refit = {"features": torch.cat(cached), "labels": pack["train"]["labels"]}
        model.head.requires_grad_(True)
        history["head_refit"] = select_state(model, batches(refit, args.batch_size, True), pack["val"],
                                             args.refit_epochs, args.head_lr, args, head=True)
        model.head.requires_grad_(False)
    # Test access is delayed until all training and validation selection finish.
    split = "val" if args.validation_only else "test"
    metrics = {"baseline": evaluate(baseline, pack[split], args.batch_size, 1, args.seed + 2000),
               "pi": evaluate(model, pack[split], args.batch_size, args.mc_samples, args.seed + 2000)}
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    source_hashes={name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest() for name in ['model.py','train.py']}
    (output / "metrics.json").write_text(json.dumps({"config": vars(args), "evaluation_split":split,"experiment_code_sha256":source_hashes,"metadata": pack.get("metadata", {}),
                                                   "metrics": metrics, "history": history}, indent=2))
    torch.save({"state_dict": model.state_dict(), "config": vars(args)}, output / "model.pt")
    print(json.dumps(metrics, indent=2))
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=["none", "fixed", "mean_only", "mean_scale", "adapter"], default="mean_scale")
    parser.add_argument("--site", choices=["feature", "concept"], default="feature")
    parser.add_argument("--conditioning", choices=["concept", "random_concept", "mlp"], default="concept")
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--refit-epochs", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--mc-samples", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--l1", type=float, default=1e-4)
    parser.add_argument("--initial-scale", type=float, default=0.01)
    parser.add_argument("--concept-weight", type=float, default=0.1)
    parser.add_argument("--energy-weight", type=float, default=0.01)
    args = parser.parse_args()
    if args.mc_samples < 1:
        parser.error("--mc-samples must be positive")
    pack = torch.load(args.pack, map_location="cpu", weights_only=True)
    run(pack, args)


if __name__ == "__main__":
    main()
