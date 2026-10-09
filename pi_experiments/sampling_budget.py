"""Compare nested Monte Carlo budgets on an unchanged selected checkpoint."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from .analyze import load_model, paired_bootstrap, task_metrics
from .prepare import atomic_save
from .resources import gpu_lease


BUDGETS = [1, 4, 8, 16]


@torch.no_grad()
def collect_budgets(model, split, seed=3310, batch_size=512):
    device = model.feature_scale.device
    predictions = {str(n): [] for n in BUDGETS}; predictions['mean'] = []
    totals = {str(n): {'standardized_concept_mean_mse': 0., 'top5_clean_agreement': 0.,
                      'sampling_top5_stability': 0., 'concept_sampling_variance': 0.} for n in BUDGETS}
    cuda = [device.index or 0] if device.type == 'cuda' else []
    with torch.random.fork_rng(devices=cuda):
        torch.manual_seed(seed)
        for features in split['features'].split(batch_size):
            features = features.to(device)
            logits, _, clean, _ = model(features, sample=False)
            predictions['mean'].append(logits.softmax(-1).cpu())
            probabilities, concepts = [], []
            for _ in range(max(BUDGETS)):
                logits, values, _, _ = model(features)
                probabilities.append(logits.softmax(-1)); concepts.append(values)
            probabilities = torch.stack(probabilities); concepts = torch.stack(concepts)
            k = min(5, clean.shape[-1]); clean_top = clean.topk(k, dim=-1).indices
            for n in BUDGETS:
                draws = concepts[:n]; average = draws.mean(0)
                top = average.topk(k, dim=-1).indices
                draw_top = draws.topk(k, dim=-1).indices
                predictions[str(n)].append(probabilities[:n].mean(0).cpu())
                metrics = {'standardized_concept_mean_mse': (average-clean).square().mean(),
                           'top5_clean_agreement': (top[:,:,None]==clean_top[:,None,:]).any(-1).float().mean(),
                           'sampling_top5_stability': (draw_top[:,:,:,None]==top[None,:,None,:]).any(-1).float().mean(),
                           'concept_sampling_variance': draws.var(0, unbiased=False).mean()}
                for key, value in metrics.items():
                    totals[str(n)][key] += value.item()*len(features)
    return {key: torch.cat(values) for key, values in predictions.items()}, {
        n: {key: value/len(split['labels']) for key, value in values.items()} for n, values in totals.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pack', required=True); parser.add_argument('--run', required=True)
    parser.add_argument('--device', default='cuda'); parser.add_argument('--seed', type=int, default=3310)
    args = parser.parse_args(); torch.set_num_threads(8)
    directory = Path(args.run); checkpoint = directory/'model.pt'
    pack = torch.load(args.pack, weights_only=True, map_location='cpu')
    state = torch.load(checkpoint, weights_only=True, map_location='cpu')
    assert not state['config'].get('validation_only') and state['config']['mode']=='mean_scale'
    with gpu_lease(f'Sampling budgets {directory}', args.device):
        model, _ = load_model(pack, checkpoint, args.device)
        predictions, diagnostics = collect_budgets(model, pack['test'], args.seed)
        del model
        if str(args.device).startswith('cuda'): torch.cuda.empty_cache()
    labels = pack['test']['labels']; predictions['labels'] = labels
    record = {'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'sampling_seed': args.seed, 'evaluation_split': 'test', 'budgets': BUDGETS,
              'protocol': 'One unchanged checkpoint; probability averages from the first n of 16 common draws in each batch; '
                          'no test selection; cached frozen image embeddings. Draws differ from earlier MC8 analysis.',
              'limits': 'Within-draw top5 stability is trivially 1 and sample variance 0 at n=1; '
                        'these are not independent repeated-run stability estimates.',
              'mean': task_metrics(predictions['mean'], labels), 'inference': {}}
    for n in BUDGETS:
        record['inference'][str(n)] = {**task_metrics(predictions[str(n)], labels), **diagnostics[str(n)],
                                      'model_forward_calls_per_batch': n,
                                      'paired_vs_mean': paired_bootstrap(predictions['mean'], predictions[str(n)], labels)}
    atomic_save(predictions, directory/'sampling_budget_predictions.pt')
    temporary = directory/'sampling_budget.partial.json'
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False)); temporary.replace(directory/'sampling_budget.json')
    print(json.dumps({key: row['accuracy'] for key, row in record['inference'].items()}), flush=True)


if __name__ == '__main__': main()
