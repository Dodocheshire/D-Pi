"""Measure anisotropy and effective rank of completed conditioning banks."""

import hashlib
import json
from pathlib import Path

import torch


def geometry(bank):
    bank = bank.double(); gram = bank@bank.T
    pairs = gram[~torch.eye(len(bank),dtype=torch.bool)]
    result = {'prototypes':len(bank),'dimension':bank.shape[1],
              'mean_direction_norm':bank.mean(0).norm().item(),
              'mean_pair_cosine':pairs.mean().item(),'pair_cosine_std':pairs.std().item()}
    for name, matrix in [('uncentered',bank),('centered',bank-bank.mean(0))]:
        eigenvalues = torch.linalg.eigvalsh(matrix@matrix.T).clamp_min(0)
        probabilities = eigenvalues/eigenvalues.sum()
        result[name+'_effective_rank'] = torch.exp(-(probabilities*probabilities.clamp_min(1e-30).log()).sum()).item()
        result[name+'_participation_rank'] = (eigenvalues.sum().square()/eigenvalues.square().sum()).item()
    return result


def main():
    torch.set_num_threads(8); rows = []
    cases = [('real','outputs/study_v1/cifar10','pi'),
             ('matched_random','outputs/conditioning_v1/cifar10','pi_random_matched'),
             ('rotated_matched','outputs/rotation_conditioning_v1/cifar10','pi_rotated_matched')]
    for variant,root,name in cases:
        for seed in range(5):
            run = Path(root)/f'{name}_seed{seed}'
            if not all((run/file).exists() for file in ['model.pt','analysis.json']):continue
            state = torch.load(run/'model.pt',weights_only=True,map_location='cpu')
            report = json.loads((run/'analysis.json').read_text())
            assert report['evaluation_split']=='test'
            metrics = report['inference']['mc8']
            rows.append({'variant':variant,'seed':seed,
                         'checkpoint_sha256':hashlib.sha256((run/'model.pt').read_bytes()).hexdigest(),
                         'geometry':geometry(state['state_dict']['generator.prototypes']),
                         'accuracy':metrics['accuracy'],
                         'learned_attention_entropy_normalized':metrics['attention_entropy_normalized']})
    result = {'dataset':'cifar10','rows':rows,
              'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'effective_rank':'exp of entropy of normalized nonnegative Gram eigenvalues (squared singular values).',
              'participation_rank':'trace(Gram)^2 / trace(Gram^2). Centered removes the mean prototype first.',
              'interpretation':'Random unit vectors also change anisotropy and effective rank. Matching initialization '
                               'does not isolate semantic alignment from geometry. Rotations preserve within-bank geometry '
                               'while changing alignment to image coordinates; trainable queries can realign.'}
    Path('reports/conditioning_v1/bank_geometry.json').write_text(json.dumps(result,indent=2))
    print('Measured conditioning-bank geometry for',len(rows),'completed checkpoints')


if __name__=='__main__':main()
