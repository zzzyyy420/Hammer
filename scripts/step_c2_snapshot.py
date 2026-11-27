"""Verify cloned/downloaded exact Step B model snapshots before C2 reuse."""
import json
import time
from pathlib import Path

from step_b_download import sha


ROOT = Path("/root/autodl-tmp/Hammer")
OUT = ROOT / "research_outputs/step_c_confirm/c2"


def main():
    for prefix, model in (("llama", "Meta-Llama-3.1-8B-Instruct"), ("qwen", "Qwen2.5-7B-Instruct")):
        metadata = ROOT / f"research_outputs/step_b_plugplay/logs/{prefix}_repository.json"
        entries = json.loads(metadata.read_bytes())["Data"]["Files"]
        files = [f for f in entries if f["Type"] == "blob" and "/" not in f["Path"]
                 and f["Path"].endswith((".json", ".safetensors", ".txt", ".model"))]
        folder = ROOT / "models" / model
        start = time.monotonic()
        while not all((folder / f["Path"]).exists() for f in files):
            assert time.monotonic() - start < 7200, "download timeout"
            time.sleep(15)
        results = []
        for entry in files:
            path = folder / entry["Path"]
            assert path.stat().st_size == entry["Size"] and sha(path) == entry["Sha256"], str(path)
            results.append({k: entry[k] for k in ("Path", "Size", "Revision", "Sha256")})
        result = {"model": model, "metadata_sha256": sha(metadata), "files": results, "verified_at_unix": time.time()}
        (OUT / f"{prefix}_snapshot.verified.json").write_text(json.dumps(result, indent=2) + "\n")
        print("SNAPSHOT_VERIFIED", model, len(results), flush=True)


if __name__ == "__main__":
    main()
