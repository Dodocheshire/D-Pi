"""Build leak-free real-data DCBM packs, with resumable SAM2 region extraction."""

import argparse
import copy
import hashlib
import json
import time
from importlib.metadata import version
import fcntl
import os
import subprocess
import sys
from pathlib import Path

import clip
import numpy as np
import torch
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder, Places365

from .model import DProjection
from .resources import gpu_lease


class ImageDataset(Dataset):
    def __init__(self, source, transform=None):
        self.source, self.transform = source, transform

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        image, label = self.source[index]
        return self.transform(image) if self.transform else image, label


def atomic_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    torch.save(value, temporary)
    temporary.replace(path)


def resize_sam_image(image, max_long_side):
    """Match the official SAM2 helper's capped OpenCV linear resize."""
    if not max_long_side or max(image.size) <= max_long_side:
        return image
    import cv2
    factor = max_long_side / max(image.size)
    size = (max(1, int(image.width*factor)), max(1, int(image.height*factor)))
    return Image.fromarray(cv2.resize(np.asarray(image.convert('RGB')), size, interpolation=cv2.INTER_LINEAR))


def load_sources(dataset, root, test_root=None):
    if dataset in {"cifar10", "cifar100"}:
        cls = CIFAR10 if dataset == "cifar10" else CIFAR100
        return cls(root, train=True, download=False), cls(root, train=False, download=False)
    if dataset == "places365":
        return Places365(root,split='train-standard',small=True,download=False), Places365(root,split='val',small=True,download=False)
    root = Path(root)
    if dataset == 'imagenet':
        train = ImageFolder(root / 'train')
        extra = ImageFolder(root / 'adapter_val')
        test = ImageFolder(test_root)
        assert train.class_to_idx == extra.class_to_idx == test.class_to_idx
        train.samples.extend(extra.samples)
        train.targets.extend(extra.targets)
        return train, test
    return ImageFolder(root / "train"), ImageFolder(root / "test")


def create_manifest(args):
    train, test = load_sources(args.dataset, args.root, args.test_root)
    labels = np.asarray(train.targets)
    train_ids, val_ids = train_test_split(np.arange(len(train)), test_size=0.1,
                                         random_state=args.split_seed, stratify=labels)
    rng = np.random.default_rng(args.split_seed + 1)
    concept_ids = []
    for label in np.unique(labels):
        candidates = train_ids[labels[train_ids] == label]
        concept_ids.extend(rng.choice(candidates, min(len(candidates), args.images_per_class), replace=False).tolist())
    manifest = {"dataset": args.dataset, "root": str(Path(args.root).resolve()),
                "backbone": args.backbone, "split_seed": args.split_seed,
                "classes": list(train.classes), "train_indices": sorted(train_ids.tolist()),
                "val_indices": sorted(val_ids.tolist()), "test_indices": list(range(len(test))),
                "concept_source_indices": sorted(concept_ids), "images_per_class": args.images_per_class,
                "official_test_root":args.test_root}
    checkpoint = Path.home() / '.cache/clip' / (args.backbone.replace('/', '-') + '.pt')
    if checkpoint.exists():
        manifest['encoder_checkpoint_sha256'] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    manifest['software'] = {name:version(name) for name in ['torch','torchvision','scikit-learn','numpy','openai-clip']}
    assert not set(manifest["train_indices"]) & set(manifest["val_indices"])
    assert set(concept_ids) <= set(manifest["train_indices"])
    return manifest


def encode_dataset(source, indices, model, transform, args):
    dataset = Subset(ImageDataset(source, transform), indices)
    loader = DataLoader(dataset, batch_size=args.encode_batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=True)
    features, labels = [], []
    started = time.monotonic()
    for i, (image, label) in enumerate(loader):
        with torch.no_grad():
            z = model.encode_image(image.to(args.device)).float().cpu()
        features.append(z)
        labels.append(label)
        if (i + 1) % 50 == 0:
            print(f"encoded {min((i+1)*args.encode_batch_size,len(dataset))}/{len(dataset)} in {time.monotonic()-started:.1f}s", flush=True)
    return {"features": torch.cat(features), "labels": torch.cat(labels), "indices": torch.tensor(indices)}


def features_stage(args, directory, manifest):
    model, transform = clip.load(args.backbone, device=args.device)
    model.eval().requires_grad_(False)
    train, test = load_sources(args.dataset, args.root, args.test_root)
    for name in ("train", "val", "test"):
        path = directory / f"{name}_features.pt"
        if path.exists():
            print("reuse", path, flush=True)
            continue
        values = encode_dataset(test if name == "test" else train, manifest[f"{name}_indices"],
                                model, transform, args)
        atomic_save(values, path)
        print("saved", path, tuple(values["features"].shape), flush=True)


def proposals_stage(args, directory, manifest):
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.build_sam import build_sam2
    if args.min_region_area:
        from sam2 import _C  # Require the configured region postprocessing before creating caches.

    sampler = build_sam2("configs/sam2.1/sam2.1_hiera_l.yaml", args.sam_checkpoint,
                         device=args.device, apply_postprocessing=False)
    generator = SAM2AutomaticMaskGenerator(sampler, points_per_side=args.points_per_side,
                                           pred_iou_thresh=0.88, stability_score_thresh=0.95,
                                           box_nms_thresh=0.5, min_mask_region_area=args.min_region_area)
    model, transform = clip.load(args.backbone, device=args.device)
    model.eval().requires_grad_(False)
    train, _ = load_sources(args.dataset, args.root, args.test_root)
    config = {"sam_checkpoint_sha256": hashlib.sha256(Path(args.sam_checkpoint).read_bytes()).hexdigest(),
              "points_per_side": args.points_per_side, "min_region_area": args.min_region_area,
              "resize_short_side": args.proposal_resolution, "backbone": args.backbone}
    config['postprocessing'] = 'sam2_cuda_connected_components' if args.min_region_area else 'none'
    if args.proposal_max_side:
        config.update(max_long_side=args.proposal_max_side, resampling='OpenCV INTER_LINEAR; floor dimensions')
    config_path = directory / "proposal_config.json"
    if config_path.exists():
        assert json.loads(config_path.read_text()) == config, "Proposal settings differ from existing cache"
    else:
        config_path.write_text(json.dumps(config, indent=2))
    shards = directory / "proposals"
    shards.mkdir(exist_ok=True)
    started = time.monotonic()
    selected = manifest["concept_source_indices"]
    if args.proposal_limit:
        selected = selected[:args.proposal_limit]
    for number, index in enumerate(selected):
        path = shards / f"{index:08d}.pt"
        if path.exists():
            continue
        image, label = train[index]
        source_image_size = list(image.size)
        if args.proposal_resolution:
            factor = args.proposal_resolution / min(image.size)
            image = image.resize((round(image.width * factor), round(image.height * factor)), Image.Resampling.BICUBIC)
        image = resize_sam_image(image, args.proposal_max_side)
        array = np.array(image.convert("RGB"))
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            masks = generator.generate(array)
        boxes = []
        crops = []
        for mask in masks:
            x, y, width, height = mask["bbox"]
            bounds = (int(x), int(y), min(image.width, int(x + width)), min(image.height, int(y + height)))
            if bounds[2] > bounds[0] and bounds[3] > bounds[1]:
                boxes.append(bounds)
                crops.append(transform(image.crop(bounds)))
        # Record failures explicitly; do not replace them with whole-image concepts.
        with torch.no_grad():
            embeddings = torch.cat([model.encode_image(torch.stack(crops[i:i+64]).to(args.device)).float().cpu()
                                    for i in range(0, len(crops), 64)]) if crops else torch.empty(0, model.visual.output_dim)
        record = {"features": embeddings, "boxes": torch.tensor(boxes).reshape(-1, 4),
                  "source_index": index, "label": label, "image_size": list(image.size)}
        if args.proposal_max_side:
            record['source_image_size'] = source_image_size
        atomic_save(record, path)
        print(f"SAM2 {number+1}/{len(selected)} image={index} regions={len(crops)} elapsed={time.monotonic()-started:.1f}s", flush=True)


def proposal_chunk_endpoints(indices,existing,chunk_size,limit=0):
    if chunk_size<=0:raise ValueError('Proposal chunk size must be positive')
    selected=indices[:limit] if limit else indices
    return [min(start+chunk_size,len(selected)) for start in range(0,len(selected),chunk_size)
            if any(index not in existing for index in selected[start:start+chunk_size])]


def proposal_chunks(args,directory,manifest):
    existing={index for index in manifest['concept_source_indices']
              if (directory/'proposals'/f'{index:08d}.pt').exists()}
    ends=proposal_chunk_endpoints(manifest['concept_source_indices'],existing,args.proposal_chunk_size,args.proposal_limit)
    for end in ends:
        command=[sys.executable,'-u','-m','pi_experiments.prepare','--dataset',args.dataset,
                 '--root',args.root,'--output',str(directory),'--stage','proposals','--proposal-worker',
                 '--proposal-limit',str(end),'--backbone',args.backbone,'--device',args.device,
                 '--split-seed',str(args.split_seed),'--images-per-class',str(args.images_per_class),
                 '--sam-checkpoint',args.sam_checkpoint,'--points-per-side',str(args.points_per_side),
                 '--min-region-area',str(args.min_region_area),'--proposal-resolution',str(args.proposal_resolution),
                 '--proposal-max-side',str(args.proposal_max_side)]
        if args.test_root:command+=['--test-root',args.test_root]
        print('SAM2 chunk through source position',end,'of',len(manifest['concept_source_indices']),flush=True)
        # Each worker exits completely, releasing model tensors and CUDA cache.
        # Atomic shards preserve identical masks/crops and interruption recovery.
        subprocess.run(command,check=True)


def fit_head(projection, data, args):
    device = args.device
    cached = {}
    for name in ("train", "val"):
        values = []
        for x in data[name]["features"].split(2048):
            with torch.no_grad():
                values.append(projection(x.to(device)))
        cached[name] = torch.cat(values)
    classes = int(data["train"]["labels"].max()) + 1
    torch.manual_seed(args.seed)
    head = torch.nn.Linear(projection.centers.shape[0], classes).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=args.head_lr)
    x, y = cached["train"], data["train"]["labels"].to(device)
    vx, vy = cached["val"], data["val"]["labels"].to(device)
    best, best_nll, history = None, float("inf"), []
    for epoch in range(args.head_epochs):
        order = torch.randperm(len(x), device=device)
        for ids in order.split(args.head_batch_size):
            optimizer.zero_grad()
            logits = head(x[ids])
            loss = F.cross_entropy(logits, y[ids]) + args.l1 * head.weight.abs().sum()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            logits = head(vx)
            nll = F.cross_entropy(logits, vy).item()
            accuracy = (logits.argmax(-1) == vy).float().mean().item()
        history.append({"epoch":epoch+1,"val_nll":nll,"val_accuracy":accuracy})
        if nll < best_nll:
            best_nll, best = nll, copy.deepcopy(head.state_dict())
        if (epoch+1) % 10 == 0:
            print(f"head epoch={epoch+1}/{args.head_epochs} val_acc={accuracy:.5f} val_nll={nll:.5f}", flush=True)
    head.load_state_dict(best)
    return head, history


def baseline_stage(args, directory, manifest):
    shards = directory / "proposals"
    needed = manifest["concept_source_indices"]
    paths = [shards / f"{index:08d}.pt" for index in needed]
    assert all(p.exists() for p in paths), "Run proposals for every manifest concept source first"
    records = [torch.load(p, weights_only=True) for p in paths]
    embeddings = torch.cat([r["features"] for r in records]).numpy()
    origins = [(r["source_index"], box.tolist()) for r in records for box in r["boxes"]]
    assert len(embeddings) >= args.clusters
    clustering = KMeans(n_clusters=args.clusters, random_state=42, n_init="auto")
    labels = clustering.fit_predict(embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True))
    representatives = []
    for label in range(args.clusters):
        ids = np.flatnonzero(labels == label)
        local = embeddings[ids]
        chosen = ids[np.abs(local - np.median(local, axis=0)).sum(1).argmin()]
        representatives.append(int(chosen))
    centers = torch.tensor(embeddings[representatives])
    data = {name: torch.load(directory / f"{name}_features.pt", weights_only=True) for name in ("train","val","test")}
    raw = (data["train"]["features"] @ centers.T / centers.square().sum(-1)).softmax(-1)
    scaler = StandardScaler().fit(raw.numpy())
    projection = DProjection(centers, torch.tensor(scaler.mean_), torch.tensor(scaler.scale_)).to(args.device)
    head, history = fit_head(projection, data, args)
    metadata = {"dataset": args.dataset, "backbone":args.backbone,"classes":manifest["classes"],
                "split_seed":args.split_seed,"baseline_seed":args.seed,"clusters":args.clusters,
                "proposal_config":json.loads((directory/'proposal_config.json').read_text()),
                "concept_sources":[origins[i] for i in representatives],
                "proposal_count":len(embeddings),"empty_proposal_images":sum(not len(r['features']) for r in records),
                "head_config":{"epochs":args.head_epochs,"batch_size":args.head_batch_size,"lr":args.head_lr,"l1":args.l1}}
    pack = {"centers":centers,"mean":torch.tensor(scaler.mean_).float(),"scale":torch.tensor(scaler.scale_).float(),
            "weight":head.weight.detach().cpu(),"bias":head.bias.detach().cpu(),"metadata":metadata,**data}
    path = directory / f"baseline_k{args.clusters}_seed{args.seed}.pt"
    atomic_save(pack, path)
    path.with_suffix('.json').write_text(json.dumps({"metadata":metadata,"history":history},indent=2))
    print("saved baseline pack",path,flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['cifar10','cifar100','cub','places365','imagenet'], required=True)
    parser.add_argument('--root',required=True)
    parser.add_argument('--test-root')
    parser.add_argument('--output',required=True)
    parser.add_argument('--stage',choices=['features','proposals','baseline'],required=True)
    parser.add_argument('--backbone',default='RN50')
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--split-seed',type=int,default=310)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--encode-batch-size',type=int,default=128)
    parser.add_argument('--images-per-class',type=int,default=50)
    parser.add_argument('--sam-checkpoint',default='artifacts/models/sam2.1_hiera_large.pt')
    parser.add_argument('--points-per-side',type=int,default=64)
    parser.add_argument('--min-region-area',type=int,default=500)
    parser.add_argument('--proposal-resolution',type=int,default=0)
    parser.add_argument('--proposal-max-side',type=int,default=None)
    parser.add_argument('--proposal-limit',type=int,default=0)
    parser.add_argument('--proposal-chunk-size',type=int,default=100)
    parser.add_argument('--proposal-worker',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--clusters',type=int,default=256)
    parser.add_argument('--head-epochs',type=int,default=200)
    parser.add_argument('--head-batch-size',type=int,default=32)
    parser.add_argument('--head-lr',type=float,default=1e-4)
    parser.add_argument('--l1',type=float,default=1e-4)
    args=parser.parse_args()
    if args.proposal_max_side is None:
        args.proposal_max_side = 1024 if args.dataset in {'places365','imagenet'} else 0
    if args.proposal_max_side < 0 or (args.proposal_max_side and args.proposal_resolution):
        parser.error('Choose either a nonnegative long-side cap or a short-side resize')
    torch.set_num_threads(8)
    directory=Path(args.output)
    directory.mkdir(parents=True,exist_ok=True)
    path=directory/'manifest.json'
    if path.exists():
        manifest=json.loads(path.read_text())
        assert manifest['dataset']==args.dataset and manifest['backbone']==args.backbone
        assert manifest['split_seed']==args.split_seed and manifest['images_per_class']==args.images_per_class
    else:
        manifest=create_manifest(args)
        path.write_text(json.dumps(manifest,indent=2))
    if args.proposal_worker:
        assert args.stage=='proposals'
        owner=json.loads((directory/'proposals.lock').read_text())
        assert owner['pid']==os.getppid(),'Proposal workers must be launched by their stage lock owner'
        with gpu_lease(f'{args.dataset}:proposals chunk',args.device):proposals_stage(args,directory,manifest)
        return
    with (directory/f'{args.stage}.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        lock.seek(0);lock.truncate()
        lock.write(json.dumps({'pid':os.getpid(),'stage':args.stage}));lock.flush()
        if args.stage=='proposals' and args.proposal_chunk_size>0:
            proposal_chunks(args,directory,manifest)
        else:
            with gpu_lease(f'{args.dataset}:{args.stage}',args.device):
                {'features':features_stage,'proposals':proposals_stage,'baseline':baseline_stage}[args.stage](args,directory,manifest)


if __name__=='__main__':
    main()
