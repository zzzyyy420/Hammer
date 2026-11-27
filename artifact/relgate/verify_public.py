"""Verify the public source, result and adapter bytes without private datasets."""
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    manifest = json.loads((HERE / 'manifest.json').read_bytes())
    assert manifest['protocol'] == 'frozen RelGate-3Way-v2, contradiction score >= 0.50'
    records = (manifest['source_files'] + manifest['template'] +
               manifest['adapters'] + manifest['published_results'])
    assert len({row.get('path', row.get('published')) for row in records}) == len(records)
    for row in records:
        relative = row.get('path', row.get('published'))
        path = ROOT / relative
        assert path.stat().st_size == row['bytes'], relative
        assert sha(path) == row['sha256'], relative
    for line in (HERE / 'RESULTS_SHA256SUMS.txt').read_text().splitlines():
        digest, path = line.split(maxsplit=1)
        assert sha(ROOT / path) == digest, path
    print('PUBLIC_ARTIFACT_VERIFIED', len(records), 'files')


if __name__ == '__main__':
    main()
