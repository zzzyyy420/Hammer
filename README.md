# Hammer and RelGate Research Artifact

This repository contains the Hammer-1.5B reproduction record and the RelGate
selective tool-name anonymization research artifact. The current artifact
entry point is [artifact/relgate/README.md](artifact/relgate/README.md).

The original Hammer checkpoint was evaluated on five official datasets with
1,252 predictions; see [the reproduction report](docs/reproduction_report.md)
for its results and limitations.

## What Is Included

- Exact experiment scripts and tests under `scripts/` and `tests/`.
- Frozen RelGate prompt, three LoRA adapters, source hashes, and a release
  verifier under `artifact/relgate/`. Adapter weights use Git LFS.
- Aggregate exploratory, independent confirmation, BFCL, and NLI comparison
  tables under `results/relgate/`. All observed positive and negative results
  are retained.
- The original Hammer and HammerBench source snapshots and their licenses under
  `upstream/`.

The repository excludes raw prompts, individual predictions, the sealed
confirmation examples, full base-model weights, and machine credentials.
The BFCL scorer is pinned to its upstream commit. See
[DATA.md](artifact/relgate/DATA.md) and
[ENVIRONMENT.md](artifact/relgate/ENVIRONMENT.md) for access and runtime details.

## Evidence Scope

The exploration pool was reused across method development; its confidence
intervals are exploratory. The independent 334-request confirmation run and
BFCL multiple/live_multiple run are reported separately. The original
Desc-Degraded confirmation annotations were unavailable, so that view is
reported as N/A. Synthetic swap labels are construction labels, not
human-verified semantic-conflict truth.


## Verify

After cloning with Git LFS enabled:

```bash
git lfs pull
python artifact/relgate/verify_public.py
python tests/test_pareto_policies.py
python scripts/p0_nli_compare.py --self-test
```

The public verifier checks all published source, adapter and result files.
`python artifact/relgate/build_release.py --check` additionally compares the
release against the locally retained frozen raw outputs when those inputs are
available.
