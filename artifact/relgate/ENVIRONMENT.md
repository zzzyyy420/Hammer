# Environment and Runtime

The runs used separate environments. Do not combine package versions from
different stages into one claimed lockfile.

| Stage | Recorded runtime |
|---|---|
| Existing Hammer reproduction and 1.5B training | Python 3.10; `environment/requirements.lock.txt` records the original installed packages, including Torch 2.5.1, Transformers 4.49.0 and PEFT 0.12.0. |
| Exploration and confirmation GPU caller runs | NVIDIA RTX 4090, BF16, vLLM 0.6.6.post1; native tokenizer chat template, `add_generation_prompt=True`; greedy generation with `do_sample=False`, `temperature=0.0`, `max_new_tokens=256`. Model-specific context and revision checks are in the run scripts and manifest. |
| NLI candidate comparison on CPU | Python 3.13.0; Torch 2.13.0+cu130 used on CPU FP32, Transformers 4.46.3, 8 CPU threads, batch size 1, maximum sequence length 512. |
| Qwen NLI caller comparison | Same frozen exploratory Qwen model, vLLM BF16 backend, chat template, maximum model length 3880, maximum 16 active sequences and batches of 128 queued prompts. Only 896 cache misses required inference. |

The GPU scripts use fixed historical paths and assume the original private
input manifests exist. A public verification run needs only Python's standard
library plus Git LFS; `tests/test_pareto_policies.py` and the NLI self-test
also need the corresponding local Python packages and scripts.

No new training or threshold search is part of the published P0-A/P0-A2
comparisons. The sole primary RelGate work point is 0.50. GPU runtime costs
and candidate-level CPU costs refer to different workloads; do not compare
their amortized latencies as online single-request latency.
