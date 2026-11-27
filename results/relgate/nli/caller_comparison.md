# P0-A2 冻结 Qwen 调用器对照

7 个探索视图，1,224 个请求视图单元；各策略使用相同的 Prompt 模板、模型、解码与评分规则，实际工具名称随策略动作变化。
历史缓存命中 1398 个唯一提示，新增推理 896 个；全部旧策略与 Step B 指标逐项一致。

新增推理报告：212.8 秒（含模型加载），JSON 解析率 95.98%，峰值显存 21.71 GiB。
## 主要观察

Target/Distractor 混合池中，RelGate 相对 NLI-Matched 的 Tool Acc 差为 +5.77 点，95% CI [+2.13, +9.62]。这是等掩码数量的端到端正向结果。
Name-Swapped 上 NLI-Matched 的点估计更高：RelGate 减 NLI-Matched 为 -3.05 点，95% CI [-8.12, +2.03]。
Verb-Flip-Target 上 NLI-0.50 的点估计更高：RelGate 减 NLI-0.50 为 -2.78 点，95% CI [-8.57, +2.60]。后两项 CI 跨零，均完整保留。

| 视图 | 策略 | Tool Acc | Strict Match | Invalid |
|---|---|---:|---:|---:|
| Clean | Keep-Name | 90.86% | 52.79% | 3.05% |
| Coarse-Desc | Keep-Name | 87.82% | 52.28% | 2.03% |
| Name-Swapped | Keep-Name | 47.72% | 28.93% | 3.05% |
| Target-Swapped | Keep-Name | 42.86% | 20.33% | 1.65% |
| Distractor-Swapped | Keep-Name | 86.81% | 52.75% | 3.30% |
| Verbose-Coarse | Keep-Name | 88.32% | 54.31% | 1.02% |
| Verb-Flip-Target | Keep-Name | 66.67% | 30.56% | 0.00% |
| Clean | Always-Mask | 81.73% | 46.70% | 4.57% |
| Coarse-Desc | Always-Mask | 60.41% | 32.99% | 1.02% |
| Name-Swapped | Always-Mask | 81.73% | 46.70% | 4.57% |
| Target-Swapped | Always-Mask | 86.81% | 51.10% | 2.75% |
| Distractor-Swapped | Always-Mask | 52.20% | 25.82% | 7.14% |
| Verbose-Coarse | Always-Mask | 51.78% | 25.89% | 7.11% |
| Verb-Flip-Target | Always-Mask | 84.72% | 48.61% | 4.17% |
| Clean | Length-64 | 84.77% | 47.72% | 3.55% |
| Coarse-Desc | Length-64 | 87.82% | 52.28% | 2.03% |
| Name-Swapped | Length-64 | 72.08% | 41.62% | 3.55% |
| Target-Swapped | Length-64 | 68.68% | 40.66% | 1.10% |
| Distractor-Swapped | Length-64 | 87.36% | 53.85% | 4.40% |
| Verbose-Coarse | Length-64 | 51.78% | 25.89% | 7.11% |
| Verb-Flip-Target | Length-64 | 73.61% | 38.89% | 0.00% |
| Clean | RelGate-0.50 | 88.32% | 53.30% | 3.05% |
| Coarse-Desc | RelGate-0.50 | 86.29% | 51.78% | 2.03% |
| Name-Swapped | RelGate-0.50 | 70.05% | 41.62% | 4.06% |
| Target-Swapped | RelGate-0.50 | 67.03% | 36.26% | 2.20% |
| Distractor-Swapped | RelGate-0.50 | 86.26% | 54.40% | 4.40% |
| Verbose-Coarse | RelGate-0.50 | 88.32% | 54.31% | 1.02% |
| Verb-Flip-Target | RelGate-0.50 | 84.72% | 44.44% | 1.39% |
| Clean | NLI-0.50 | 87.82% | 49.75% | 3.05% |
| Coarse-Desc | NLI-0.50 | 80.20% | 47.72% | 2.54% |
| Name-Swapped | NLI-0.50 | 68.02% | 39.09% | 3.55% |
| Target-Swapped | NLI-0.50 | 60.99% | 31.87% | 2.75% |
| Distractor-Swapped | NLI-0.50 | 82.42% | 50.00% | 3.30% |
| Verbose-Coarse | NLI-0.50 | 78.17% | 46.19% | 3.55% |
| Verb-Flip-Target | NLI-0.50 | 87.50% | 45.83% | 1.39% |
| Clean | NLI-Matched | 87.82% | 49.75% | 3.05% |
| Coarse-Desc | NLI-Matched | 85.79% | 51.27% | 2.03% |
| Name-Swapped | NLI-Matched | 73.10% | 44.16% | 4.06% |
| Target-Swapped | NLI-Matched | 59.34% | 31.32% | 2.20% |
| Distractor-Swapped | NLI-Matched | 82.42% | 52.20% | 3.30% |
| Verbose-Coarse | NLI-Matched | 88.32% | 54.31% | 1.02% |
| Verb-Flip-Target | NLI-Matched | 84.72% | 44.44% | 1.39% |

## 联合配对 Bootstrap

20,000 次、按原始请求 ID 在七视图和策略间联合重抽；95% 百分位 CI。以下为 RelGate 减 NLI，单位百分点；未做多重比较校正。

| 指标 | 对照 | 视图 | 差值 | 95% CI |
|---|---|---|---:|---:|
| tool_acc | RelGate-0.50 minus NLI-0.50 | Clean | +0.51 | [-2.54, +4.06] |
| tool_acc | RelGate-0.50 minus NLI-0.50 | Coarse-Desc | +6.09 | [+1.52, +10.66] |
| tool_acc | RelGate-0.50 minus NLI-0.50 | Name-Swapped | +2.03 | [-4.06, +7.61] |
| tool_acc | RelGate-0.50 minus NLI-0.50 | Target-Swapped | +6.04 | [+0.00, +12.29] |
| tool_acc | RelGate-0.50 minus NLI-0.50 | Distractor-Swapped | +3.85 | [-0.56, +8.62] |
| tool_acc | RelGate-0.50 minus NLI-0.50 | Verbose-Coarse | +10.15 | [+6.09, +14.21] |
| tool_acc | RelGate-0.50 minus NLI-0.50 | Verb-Flip-Target | -2.78 | [-8.57, +2.60] |
| tool_acc | RelGate-0.50 minus NLI-0.50 | Target+Distractor | +4.95 | [+1.06, +9.07] |
| tool_acc | RelGate-0.50 minus NLI-Matched | Clean | +0.51 | [-2.54, +3.55] |
| tool_acc | RelGate-0.50 minus NLI-Matched | Coarse-Desc | +0.51 | [-2.03, +3.05] |
| tool_acc | RelGate-0.50 minus NLI-Matched | Name-Swapped | -3.05 | [-8.12, +2.03] |
| tool_acc | RelGate-0.50 minus NLI-Matched | Target-Swapped | +7.69 | [+2.26, +13.19] |
| tool_acc | RelGate-0.50 minus NLI-Matched | Distractor-Swapped | +3.85 | [-0.56, +8.70] |
| tool_acc | RelGate-0.50 minus NLI-Matched | Verbose-Coarse | +0.00 | [+0.00, +0.00] |
| tool_acc | RelGate-0.50 minus NLI-Matched | Verb-Flip-Target | +0.00 | [+0.00, +0.00] |
| tool_acc | RelGate-0.50 minus NLI-Matched | Target+Distractor | +5.77 | [+2.13, +9.62] |
| strict_match | RelGate-0.50 minus NLI-0.50 | Clean | +3.55 | [+0.00, +7.61] |
| strict_match | RelGate-0.50 minus NLI-0.50 | Coarse-Desc | +4.06 | [+0.51, +7.61] |
| strict_match | RelGate-0.50 minus NLI-0.50 | Name-Swapped | +2.54 | [-2.03, +7.61] |
| strict_match | RelGate-0.50 minus NLI-0.50 | Target-Swapped | +4.40 | [-1.69, +10.81] |
| strict_match | RelGate-0.50 minus NLI-0.50 | Distractor-Swapped | +4.40 | [+0.00, +8.99] |
| strict_match | RelGate-0.50 minus NLI-0.50 | Verbose-Coarse | +8.12 | [+4.57, +12.18] |
| strict_match | RelGate-0.50 minus NLI-0.50 | Verb-Flip-Target | -1.39 | [-4.62, +0.00] |
| strict_match | RelGate-0.50 minus NLI-0.50 | Target+Distractor | +4.40 | [+0.55, +8.42] |
| strict_match | RelGate-0.50 minus NLI-Matched | Clean | +3.55 | [+0.51, +7.11] |
| strict_match | RelGate-0.50 minus NLI-Matched | Coarse-Desc | +0.51 | [-2.03, +3.05] |
| strict_match | RelGate-0.50 minus NLI-Matched | Name-Swapped | -2.54 | [-6.60, +1.52] |
| strict_match | RelGate-0.50 minus NLI-Matched | Target-Swapped | +4.95 | [+0.00, +10.11] |
| strict_match | RelGate-0.50 minus NLI-Matched | Distractor-Swapped | +2.20 | [-2.17, +6.56] |
| strict_match | RelGate-0.50 minus NLI-Matched | Verbose-Coarse | +0.00 | [+0.00, +0.00] |
| strict_match | RelGate-0.50 minus NLI-Matched | Verb-Flip-Target | +0.00 | [+0.00, +0.00] |
| strict_match | RelGate-0.50 minus NLI-Matched | Target+Distractor | +3.57 | [+0.00, +7.14] |

## 不一致请求观察

下表仅计两种门控掩码向量不同的请求；金标掩码与配对预测是观察性描述，不作因果归因。

| 视图 | NLI 策略 | 不一致请求 | RelGate 独对 | NLI 独对 | 双对 | 双错 |
|---|---|---:|---:|---:|---:|---:|
| Name-Swapped | NLI-0.50 | 152 | 19 | 15 | 83 | 35 |
| Name-Swapped | NLI-Matched | 94 | 10 | 16 | 41 | 27 |
| Verb-Flip-Target | NLI-0.50 | 19 | 1 | 3 | 15 | 0 |
| Verb-Flip-Target | NLI-Matched | 3 | 0 | 0 | 3 | 0 |

逐请求金标名称掩码状态、配对成功/失败见 `discordant_requests.csv`；交叉计数见 `discordance_summary.csv`。
截断敏感性复用 P0-A 的 `truncation_by_view.csv` 与原候选分数；本轮未重新打分。
本轮只使用冻结探索视图，未读取 334 条独立确认集；未训练、调阈值、改模板或扩展调用器。
