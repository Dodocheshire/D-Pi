"""Numerical agreement with upstream preprocessing and Pi training invariants."""

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from .model import DPi, DProjection, export_cbm


def upstream_preprocess():
    # Execute the unchanged upstream method without loading its unrelated dependencies.
    source = Path(__file__).parents[1] / "utils/dcbm.py"
    tree = ast.parse(source.read_text())
    cbm = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "CBM")
    method = next(node for node in cbm.body if isinstance(node, ast.FunctionDef) and node.name == "preprocess_module")
    namespace = {"torch": torch, "np": np, "tqdm": lambda x, **kwargs: x}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
    return namespace["preprocess_module"]


def fixture():
    torch.manual_seed(4)
    centers, features = torch.randn(7, 6), torch.randn(19, 6)
    raw = upstream_preprocess()(SimpleNamespace(device="cpu"), features.numpy(), centers.numpy())
    scaler = StandardScaler().fit(raw)
    projection = DProjection(centers, torch.tensor(scaler.mean_), torch.tensor(scaler.scale_))
    return centers, features, scaler, projection


def test_projection_matches_official_function_and_scaler():
    centers, features, scaler, projection = fixture()
    raw = upstream_preprocess()(SimpleNamespace(device="cpu"), features.numpy(), centers.numpy(), batch_size=3)
    np.testing.assert_allclose(projection(features).numpy(), scaler.transform(raw), rtol=2e-5, atol=2e-5)


def test_zero_noise_matches_official_logits():
    _, features, _, projection = fixture()
    weight, bias = torch.randn(3, 7), torch.randn(3)
    model = DPi(projection, weight, bias, torch.ones(6), mode="none")
    logits, concepts, clean, energy = model(features)
    torch.testing.assert_close(logits, clean @ weight.T + bias)
    torch.testing.assert_close(concepts, clean)
    assert energy == 0


def test_noise_gradient_and_frozen_semantics():
    for site in ("feature", "concept"):
        _, features, _, projection = fixture()
        model = DPi(projection, torch.randn(3, 7), torch.randn(3), torch.ones(6), site=site)
        before = projection.centers.clone()
        logits, concepts, clean, energy = model(features)
        loss = torch.nn.functional.cross_entropy(logits, torch.arange(len(features)) % 3)
        loss = loss + 0.1 * (concepts - clean).square().mean() + 0.01 * energy
        loss.backward()
        assert model.generator.mean.weight.grad.abs().sum() > 0
        assert model.generator.log_scale.weight.grad.abs().sum() > 0
        assert model.head.weight.grad is None
        torch.testing.assert_close(before, projection.centers)


def test_export_preserves_official_order(tmp_path):
    centers, features, scaler, projection = fixture()
    head = SimpleNamespace(linear=torch.nn.Linear(7, 3))
    official = SimpleNamespace(clustered_concepts=centers.numpy(), scaler=scaler, model=head,
                               dataset="synthetic", clip_model="synthetic")
    for split in ("train", "val", "test"):
        setattr(official, f"data_{split}", features.numpy())
        setattr(official, f"{split}_labels_int", np.arange(len(features)) % 3)
    path = tmp_path / "pack.pt"
    export_cbm(official, path)
    pack = torch.load(path, weights_only=True)
    torch.testing.assert_close(pack["centers"], centers)
    restored = DProjection(pack["centers"], pack["mean"], pack["scale"])
    torch.testing.assert_close(restored(features), projection(features))


def test_matched_adapter_has_same_trainable_capacity_and_no_sampling_effect():
    centers, features, _, projection = fixture()
    weight, bias = torch.randn(3,7),torch.randn(3)
    gaussian=DPi(projection,weight,bias,torch.ones(6),mode='mean_scale')
    adapter=DPi(projection,weight,bias,torch.ones(6),mode='adapter')
    count=lambda model:sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert count(gaussian)==count(adapter)
    torch.testing.assert_close(adapter(features,sample=True)[0],adapter(features,sample=False)[0])


def test_noise_decomposition_preserves_checkpoint_and_removes_sampling():
    from .noise_diagnostics import NoiseOverride

    _,features,_,projection=fixture()
    model=DPi(projection,torch.randn(3,7),torch.randn(3),torch.ones(6),mode='mean_scale')
    original=model.generator
    with torch.no_grad():original.mean.weight.normal_(std=0.01)
    state={k:v.clone() for k,v in original.state_dict().items()}
    expected=model(features,sample=False)[0]
    model.generator=NoiseOverride(original,scale_multiplier=0)
    torch.testing.assert_close(model(features,sample=True)[0],expected)
    model.generator=NoiseOverride(original,mean_multiplier=0,scale_multiplier=0)
    torch.testing.assert_close(model(features)[0],model.head(projection(features)))
    for key,value in original.state_dict().items():torch.testing.assert_close(value,state[key])


def test_validation_search_does_not_access_test_split(tmp_path):
    from .train import run

    class GuardedPack(dict):
        def __getitem__(self,key):
            assert key!='test','validation search accessed the held-out test split'
            return super().__getitem__(key)

    centers,features,_,projection=fixture()
    split={'features':features,'labels':torch.arange(len(features))%3}
    pack=GuardedPack(centers=centers,mean=projection.mean,scale=projection.scale,
                     weight=torch.randn(3,7),bias=torch.randn(3),train=split,val=split)
    args=SimpleNamespace(seed=0,mode='mean_scale',conditioning='concept',site='feature',hidden=8,
                         initial_scale=0.01,device='cpu',batch_size=8,epochs=1,lr=3e-4,
                         concept_weight=0.1,energy_weight=0.01,mc_samples=2,refit_epochs=0,
                         validation_only=True,output=str(tmp_path/'search'))
    run(pack,args)
