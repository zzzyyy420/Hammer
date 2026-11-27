# RelGate Reproducibility Artifact

This directory indexes the code and evidence for frozen RelGate-3Way-v2 at
the single primary threshold `CONTRADICTED >= 0.50`. It is a record of executed
experiments, including negative results. It does not present a new threshold,
new training run, or a retuned NLI template.

## Repository Layout

| Location | Contents |
|---|---|
| `scripts/` | Original data construction, gate training/scoring, caller inference, official BFCL scoring, bootstrap and NLI comparison code. Original filenames remain stable. |
| `tests/` | Frozen data audits and data-independent policy checks. |
| `artifact/relgate/templates/` | Exact three-label gate instruction. |
| `artifact/relgate/weights/` | Exact LoRA adapters for RelGate-3Way-v2, Hammer-Repro 1.5B and B2 1.5B, stored with Git LFS. |
| `results/relgate/exploration/` | Eight-view metrics and paired statistics. |
| `results/relgate/confirmation/` | Independent 334-request seven-view and official BFCL AST aggregates. |
| `results/relgate/nli/` | Candidate-level NLI and Qwen2.5-7B end-to-end NLI comparisons, including `caller_comparison.md`. |
| `manifest.json` | Source, result, adapter and template SHA-256 values with local source paths. |

The original Hammer and HammerBench snapshots stay in `upstream/` with their
own licenses. The `paper/draft_acl` manuscript is an earlier exploratory draft
and is excluded because its pending-experiment statements are now outdated.

## Verify the Published Artifact

```bash
git lfs pull
python artifact/relgate/verify_public.py
python tests/test_pareto_policies.py
python scripts/p0_nli_compare.py --self-test
```

`verify_public.py` needs no research data, model runtime, or GPU. It verifies
every listed source, result, template and adapter against `manifest.json`.
The `build_release.py --check` command additionally compares the public copies
with the frozen local `research_outputs/` originals. The latter are not in Git.

## Reproduction Map

The scripts are preserved as executed, with `research_outputs/` as their
artifact store. They are stage-specific research entry points, not a generic
Python package. Some GPU scripts expect the project at
`/root/autodl-tmp/Hammer`; use the same directory layout for an exact rerun.
See [ENVIRONMENT.md](ENVIRONMENT.md) and [DATA.md](DATA.md) before running them.

| Stage | Entry points | Published evidence |
|---|---|---|
| Frozen exploration and baselines | `build_relgate_dataset.py`, `build_relgate_v2.py`, `step_a_probes.py`, `step_b_prepare.py`, `step_b_vllm_runner.py`, `step_b_analyze.py` | `results/relgate/exploration/` |
| Independent confirmation and BFCL | `step_c1_prepare.py`, `step_c15_prepare.py`, `step_c2_freeze.py`, `step_c2_score.py`, `step_c2_schedule.py`, `step_c2_caller.py`, `step_c2_analyze.py` | `results/relgate/confirmation/` |
| NLI candidate and caller comparison | `p0_nli_prepare.py`, `p0_nli_compare.py`, `p0_a2_prepare.py`, `p0_a2_caller.py`, `p0_a2_analyze.py` | `results/relgate/nli/` |

`step_c_bfcl_adapter.py` uses the official BFCL AST scorer from the pinned
external source. `Length-64` remains in every main comparison. The P0-A2 NLI
caller comparison uses Qwen2.5-7B only and reuses Step B prompt predictions
when the full prompt and model configuration match.

## Interpreting the Tables

- Exploration uses a repeatedly consulted 197-request pool. Pointwise
  bootstrap intervals are descriptive and do not turn this pool into an
  independent test set.
- Confirmation uses the original independent 334-request split. Its
  Desc-Degraded annotation list was unavailable; no substitute split was
  sampled and that view remains N/A.
- BFCL results use official `multiple` and `live_multiple` answers and AST
  scoring. Original, name-swapped, and frozen verb-flip views are separate.
- Name-swap and verb-flip construction flags are not human-verified semantic
  conflict labels. Tool Accuracy, argument-level Strict Match, and official
  BFCL AST accuracy are distinct metrics.
- NLI-0.50 and per-request equal-dose NLI-Matched are both reported. On
  Target/Distractor mixed exploration views, RelGate minus NLI-Matched Tool
  Accuracy is +5.77 percentage points with 95% paired CI [+2.13, +9.62].
  NLI-Matched has a higher Name-Swapped point estimate, and NLI-0.50 has a
  higher Verb-Flip point estimate; their paired intervals cross zero.

The public tables keep all policies and callers, including results where
RelGate is worse. Raw prompts, per-request outputs, and the private
confirmation examples are not in this repository.
