# Hammer 1.5B checkpoint evaluation report

Status: partially_reproduced. Five released academic evaluation pipelines completed; retraining, BFCL and ablations were not executed.

| Dataset | N | F1_api percent | F1_param percent |
|---|---:|---:|---:|
| apibank_l1 | 399 | 83.38 | 73.90 |
| apibank_l2 | 127 | 77.88 | 60.50 |
| NexusRaven | 318 | 80.46 | 59.34 |
| sealtool | 294 | 95.06 | 89.46 |
| toolalpaca | 114 | 80.75 | 53.39 |

Macro F1_api: 83.51; macro F1_param: 67.32.

## Interpretation

Author evaluate.py metrics are reported using their raw names; these are not exact joint tool-call accuracy. Paper Table 3 is a reference, not proof of exact protocol parity. NexusRaven F1_param differs by +2.46 percentage points, outside the pre-recorded 2pp tolerance. ToolAlpaca release has 114 rows versus 100 mentioned in paper text. Do not claim full numerical paper reproduction.

## Environment and engineering deviations

Python 3.10.22, torch 2.5.1 CUDA 12.4, transformers 4.49.0, PEFT 0.12.0, datasets 3.2.0, accelerate 1.2.1, TRL 0.9.6, DeepSpeed 0.16.4 and LLaMA-Factory 0.9.2. Dependencies pass pip check. GPU BF16, Qwen2 PEFT forward and FusedAdam build/load pass. An independent adapter changes --dataset to --eval_dataset; original author code remains unchanged. CUDA header search path and metrics extras were added. Raw failed-attempt logs are retained.

## Assets

Original checkpoint remains in /root/autodl-tmp/Hammer/models/Hammer-1.5b. No newly trained checkpoint or optimizer state exists because training was not started. Full raw logs, predictions, effective commands and source manifests remain with this report.

## Training data

load_dataset("Salesforce/xlam-function-calling-60k") was attempted. Official domain timed out; mirror returned gated-dataset authorization failure. Obtain authorized access before training.