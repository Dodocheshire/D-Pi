"""Record live process identities, completed artifacts and implementation checks."""

import hashlib
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET

from .suite import pid_identity


def main():
    official = json.loads(Path('UPSTREAM.json').read_text())['official_file_sha256']
    changed = [name for name, digest in official.items()
               if not Path(name).exists() or hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest]
    assert not changed, changed
    tests = ET.parse('reports/IMPLEMENTATION_CHECKS.xml').getroot().find('testsuite').attrib
    managers = {}
    for path in sorted(Path('artifacts/logs').rglob('status.json')):
        state = json.loads(path.read_text())
        pid = state.get('manager_pid')
        if pid is None:
            continue
        child = state.get('child_pid')
        managers[str(path.parent.relative_to('artifacts/logs'))] = {
            'pid': pid, 'alive': pid_identity(pid) is not None, 'process_start_ticks': pid_identity(pid),
            'phase': state.get('phase'), 'child_pid': child,
            'child_alive': pid_identity(child) is not None if child else False,
            'completed_commands': len(state.get('completed_commands', [])),
            'source_status': str(path)}
    datasets = {}
    for path in sorted(Path('artifacts/datasets').glob('*/manifest.json')):
        directory = path.parent
        manifest = json.loads(path.read_text())
        indices = manifest.get('concept_source_indices', [])
        datasets[directory.name] = {
            'split_sizes': {split: len(manifest[f'{split}_indices']) for split in ['train', 'val', 'test']
                            if f'{split}_indices' in manifest},
            'proposals_complete': sum((directory / 'proposals' / f'{index:08d}.pt').exists() for index in indices),
            'proposals_required': len(indices),
            'feature_files': {split: (directory / f'{split}_features.pt').exists()
                              for split in ['train', 'val', 'test'] if f'{split}_indices' in manifest}}
    owner_path = Path('artifacts/resources/gpu_owner.json')
    owner = json.loads(owner_path.read_text()) if owner_path.exists() else {}
    owner['owner_alive'] = pid_identity(owner['owner_pid']) is not None if owner.get('owner_pid') else False
    record = {'checked_at': time.time(), 'official_files_unchanged': len(official),
              'implementation_tests': tests, 'managers': managers, 'gpu_lease': owner,
              'new_entry_code_sha256': {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                        for path in sorted(Path('pi_experiments').glob('*.py'))}}
    preparation = {'checked_at': record['checked_at'], 'datasets': datasets,
                   'official_files_verified': len(official), 'checks_passed': int(tests['tests']) - int(tests['failures']) - int(tests['errors']),
                   'additional_studies': managers,
                   'cub_annotations': {'attributes': 312, 'parts': 15, 'primary_min_certainty': 3,
                                       'sensitivity_min_certainties': [2, 4]}}
    for name, values in [('RUNNING_EXPERIMENTS.json', record), ('PREPARATION_STATUS.json', preparation)]:
        path = Path('reports') / name
        temporary = path.with_suffix('.partial.json')
        temporary.write_text(json.dumps(values, indent=2))
        temporary.replace(path)
    print(json.dumps({'checked_at': record['checked_at'], 'official_files': len(official), 'tests': tests['tests'],
                      'live_managers': sum(row['alive'] for row in managers.values()), 'gpu_lease': owner}, indent=2))


if __name__ == '__main__':
    main()
