"""Differentiable counterpart of utils/dcbm.py:preprocess_module/preprocess_data."""

import math

import torch
from torch import nn
from torch.nn import functional as F


class DProjection(nn.Module):
    def __init__(self, centers, mean, scale):
        super().__init__()
        self.register_buffer("centers", centers.float().clone())
        self.register_buffer("mean", mean.float().clone())
        self.register_buffer("scale", scale.float().clone())

    def scores(self, features):
        return features @ self.centers.T / self.centers.square().sum(-1)

    def standardize(self, scores):
        # The official implementation adds softmax to the paper's Eq. 2.
        return (scores.softmax(-1) - self.mean) / self.scale

    def forward(self, features):
        return self.standardize(self.scores(features))


class PiGenerator(nn.Module):
    def __init__(self, dimension, centers, hidden=128, initial_scale=0.01,
                 conditioning="concept", mode="mean_scale", output_dim=None):
        super().__init__()
        self.conditioning = conditioning
        self.mode = mode
        self.initial_scale = initial_scale
        prototypes = torch.randn_like(centers.float()) if conditioning == "random_concept" else centers.float()
        self.register_buffer("prototypes", F.normalize(prototypes, dim=-1))
        self.query = nn.Linear(dimension, dimension, bias=False) if conditioning in {"concept","random_concept"} else None
        width = 2 * dimension if self.query is not None else dimension
        self.net = nn.Sequential(nn.Linear(width, hidden), nn.GELU())
        self.mean = nn.Linear(hidden, output_dim or dimension)
        self.log_scale = nn.Linear(hidden, output_dim or dimension)
        nn.init.zeros_(self.mean.weight)
        nn.init.zeros_(self.mean.bias)
        nn.init.zeros_(self.log_scale.weight)
        nn.init.constant_(self.log_scale.bias, math.log(math.expm1(initial_scale)))
        if mode == "mean_only":
            self.log_scale.requires_grad_(False)

    def parameters_for(self, features):
        context = F.normalize(features, dim=-1)
        if self.query is not None:
            weights = (self.query(context) @ self.prototypes.T).softmax(-1)
            context = torch.cat([context, weights @ self.prototypes], dim=-1)
        hidden = self.net(context)
        mean = self.mean(hidden)
        if self.mode == "adapter":
            mean = mean + F.softplus(self.log_scale(hidden)) - self.initial_scale
        scale = F.softplus(self.log_scale(hidden)) if self.mode == "mean_scale" else torch.zeros_like(mean)
        return mean, scale


class DPi(nn.Module):
    def __init__(self, projection, weight, bias, feature_scale, mode="mean_scale",
                 conditioning="concept", site="feature", hidden=128, initial_scale=0.01):
        super().__init__()
        self.projection = projection
        self.site = site
        self.mode = mode
        self.initial_scale = initial_scale
        dimension = projection.centers.shape[1]
        self.head = nn.Linear(projection.centers.shape[0], weight.shape[0])
        with torch.no_grad():
            self.head.weight.copy_(weight)
            self.head.bias.copy_(bias)
        self.head.requires_grad_(False)
        self.register_buffer("feature_scale", feature_scale.float())
        self.generator = None
        if mode in {"mean_scale", "mean_only", "adapter"}:
            self.generator = PiGenerator(dimension, projection.centers, hidden,
                                         initial_scale, conditioning, mode,
                                         dimension if site == "feature" else projection.centers.shape[0])

    def forward(self, features, sample=True):
        clean = self.projection(features)
        scores = self.projection.scores(features)
        width = features.shape[-1] if self.site == "feature" else scores.shape[-1]
        if self.generator is None:
            mean = features.new_zeros(features.shape[0], width)
            scale = torch.full_like(mean, self.initial_scale) if self.mode == "fixed" else torch.zeros_like(mean)
        else:
            mean, scale = self.generator.parameters_for(features)
        noise = mean + scale * torch.randn_like(scale) if sample else mean
        if self.site == "feature":
            concepts = self.projection(features + self.feature_scale * noise)
        else:
            concepts = self.projection.standardize(scores + noise)
        logits = self.head(concepts)
        # Analytic expected energy gives the variance branch a stable gradient.
        energy = (mean.square() + scale.square()).mean()
        return logits, concepts, clean, energy


def export_cbm(cbm, path):
    """Call after an official CBM is fitted; export tensors in its existing order."""
    pack = {
        "centers": torch.as_tensor(cbm.clustered_concepts, dtype=torch.float32),
        "mean": torch.as_tensor(cbm.scaler.mean_, dtype=torch.float32),
        "scale": torch.as_tensor(cbm.scaler.scale_, dtype=torch.float32),
        "weight": cbm.model.linear.weight.detach().cpu(),
        "bias": cbm.model.linear.bias.detach().cpu(),
        "metadata": {"dataset": cbm.dataset, "backbone": cbm.clip_model,
                     "projection": "dot_over_norm_squared_then_softmax_then_standard_scaler"},
    }
    for split in ("train", "val", "test"):
        pack[split] = {
            "features": torch.as_tensor(getattr(cbm, f"data_{split}"), dtype=torch.float32).reshape(-1, pack["centers"].shape[1]),
            "labels": torch.as_tensor(getattr(cbm, f"{split}_labels_int"), dtype=torch.long),
        }
    torch.save(pack, path)
