"""Resume ModelScope files and verify every file against repository metadata."""
import argparse
import concurrent.futures
import hashlib
import json
import subprocess
from pathlib import Path

import requests


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True)
    p.add_argument("--dest", type=Path, required=True)
    p.add_argument("--record", type=Path, required=True)
    a = p.parse_args()
    url = f"https://modelscope.cn/api/v1/models/{a.repo}/repo/files"
    response = requests.get(url, params={"Revision": "master", "Recursive": "true"}, timeout=30)
    response.raise_for_status()
    meta = response.json()
    assert meta["Success"], meta
    a.dest.mkdir(parents=True, exist_ok=True)
    a.record.parent.mkdir(parents=True, exist_ok=True)
    a.record.write_text(json.dumps(meta, indent=2) + "\n")
    files = [f for f in meta["Data"]["Files"] if f["Type"] == "blob"
             and not f["Path"].endswith((".bin", ".pt", ".pth"))]

    def fetch(f):
        target = a.dest / f["Path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            assert target.stat().st_size == f["Size"] and sha(target) == f["Sha256"], target
            print("verified", f["Path"], flush=True)
            return
        part = target.with_name(target.name + ".incomplete")
        endpoint = f"https://modelscope.cn/models/{a.repo}/resolve/{f['Revision']}/{f['Path']}"
        subprocess.run(["curl", "-fL", "--retry", "8", "--retry-delay", "2", "--connect-timeout", "20",
                        "--speed-limit", "1024", "--speed-time", "90", "-C", "-", "-o", str(part), endpoint], check=True)
        assert part.stat().st_size == f["Size"] and sha(part) == f["Sha256"], target
        part.rename(target)
        print("downloaded_verified", f["Path"], flush=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(fetch, files))
    print("ALL_FILES_VERIFIED", a.repo, flush=True)


if __name__ == "__main__":
    main()
