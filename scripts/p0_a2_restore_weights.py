"""Restore the exact four Step B Qwen weight shards on the remote host."""
import concurrent.futures
import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path('/root/autodl-tmp/Hammer')
OUT = ROOT / 'research_outputs/p0_a2'
DEST = ROOT / 'models/Qwen2.5-7B-Instruct'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def fetch(row):
    target = DEST / row['path']
    if target.exists():
        assert sha(target) == row['sha256'], target
        print('VERIFIED_EXISTING', target.name, flush=True)
        return
    part = target.with_suffix(target.suffix + '.incomplete')
    url = f"https://modelscope.cn/models/Qwen/Qwen2.5-7B-Instruct/resolve/{row['revision']}/{row['path']}"
    subprocess.run(['curl', '-fL', '--retry', '8', '--retry-delay', '3', '--connect-timeout', '20',
                    '--speed-limit', '1024', '--speed-time', '120', '-C', '-', '-o', str(part), url], check=True)
    assert sha(part) == row['sha256'], target
    part.rename(target)
    print('DOWNLOADED_VERIFIED', target.name, target.stat().st_size, flush=True)


def main():
    provenance = json.loads((OUT / 'step_b_model_provenance.json').read_bytes())['Qwen2.5-7B-Instruct']
    assert provenance['repository'] == 'Qwen/Qwen2.5-7B-Instruct'
    assert len(provenance['weight_files']) == 4
    DEST.mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(fetch, provenance['weight_files']))
    print('ALL_FROZEN_QWEN_WEIGHTS_VERIFIED', flush=True)


if __name__ == '__main__':
    main()
