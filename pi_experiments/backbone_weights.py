"""Fetch official OpenAI CLIP weights and record the URL-embedded hash."""

import hashlib
import json
from pathlib import Path

from clip.clip import _MODELS, _download


def main():
    directory = Path('artifacts/models/clip')
    directory.mkdir(parents=True, exist_ok=True)
    for name in ['ViT-B/16', 'ViT-L/14']:
        url = _MODELS[name]
        path = Path(_download(url, str(Path.home() / '.cache/clip')))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        expected = url.split('/')[-2]
        assert digest == expected
        record = {'backbone': name, 'source': url, 'path': str(path), 'sha256': digest,
                  'expected_sha256': expected, 'verified': True}
        (directory / (name.replace('/', '') + '.json')).write_text(json.dumps(record, indent=2))
        print('Verified official CLIP weights', name, digest, flush=True)


if __name__ == '__main__':
    main()
