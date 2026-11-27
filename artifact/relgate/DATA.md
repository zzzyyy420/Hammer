# Data and Model Inputs

The public repository contains code, three LoRA adapters, aggregate results,
and SHA-256 provenance. It does not redistribute the training corpus,
individual evaluation requests, or the original 334-request confirmation
split. No alternate random split should be substituted for that split.

## External Inputs

| Input | Identity / use |
|---|---|
| xLAM function-calling 60K | Source data for the recorded gate and caller training. Obtain under the source dataset's terms. Dataset construction and disjointness checks are in `build_relgate_dataset.py` and `build_relgate_v2.py`. |
| BFCL | Official BFCL source commit `6ea57973c7a6097fd7c5915698c54c17c5b1b6c8`; use its `multiple` and `live_multiple` data, possible answers, and AST scorer together. |
| Qwen2-1.5B-Instruct | Base for the three published LoRA adapters. The local reproduction record pins its base revision in `configs/hammer15b_train_v1/base_model_revision.json`. |
| Qwen/Qwen2.5-7B-Instruct | Frozen exploratory and NLI-comparison caller. Weight revisions and file hashes are in `reference/caller_model_provenance.json`. |
| LLM-Research/Meta-Llama-3.1-8B-Instruct | Frozen cross-family caller for exploration and confirmation. Obtain under its model terms. |
| cross-encoder/nli-deberta-v3-large | P0-A comparator, pinned to Hugging Face commit `bab4bc7178836f731dcfd18c06ca9def0a137712`; CPU FP32. |

The three published adapter files are versioned with Git LFS. The adapter
configuration files retain the historical machine-specific
`base_model_name_or_path`; resolve that path to the exact downloaded base
model in a new environment. Adapter SHA-256 values are in `manifest.json`.

The frozen exploration probes, candidate schedules, held-out split metadata,
and raw output hashes are recorded in the local `research_outputs/` manifests.
The release copies only aggregate tables, so a public checkout can verify the
published evidence but cannot reconstruct private per-request predictions.

## Split and Metric Boundaries

- Exploration: 197 Clean/Coarse/Name-Swapped/Verbose-Coarse requests, 182
  Target/Distractor requests, and 72 Verb-Flip-Target requests in P0-A2.
- Independent confirmation: 334 original requests; 309 each for the eligible
  Target/Distractor views and 156 Verb-Flip-Target hits. Desc-Degraded is N/A.
- BFCL: 200 `multiple` and 1,053 `live_multiple` requests; the frozen verb
  vocabulary hits 177 targets across both splits.

Reuse the same original request ID across paired views and strategies when
resampling. The published confidence intervals are pointwise 95% percentile
intervals with 20,000 joint paired draws, without multiplicity correction.
