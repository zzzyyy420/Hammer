# Hammer 复现项目

本项目复现 Hammer-1.5B 的官方权重评测，后续在完整训练数据获批后，从原始 Qwen2-1.5B-Instruct 基座重训。当前未宣称论文结果已复现。

## 当前进度

- 本地包上传服务器并解压：完成，62 个文件通过 SHA-256 清单校验。
- 环境：独立 Conda Python 3.10；依赖安装完成，pip check、BF16 运算、Qwen2＋PEFT 前向和 DeepSpeed FusedAdam 编译加载均通过。
- 官方评测集：API-Bank L1 399、L2 127、NexusRaven 318、Seal-Tools 294、Tool-Alpaca 114，共 1252 条。
- 官方权重：Hammer-1.5B 的五个评测集已完成，共 1252 条预测；预测数量、JSON 格式和产物 SHA-256 校验通过。
- 重新训练：未启动。用户决定先完成 checkpoint 评测；60K 训练数据加载已尝试，官方域名超时，镜像返回 gated dataset，需获批访问权限。

## 目录

- `upstream/Hammer`：作者代码副本，保留原始 LICENSE。数据和大文件不进入 Git。
- `upstream/HammerBench`：作者补充 benchmark 代码，和主论文五个评测集分开。
- `environment`：服务器环境安装脚本与检查脚本；完整日志保存在服务器及本地电脑，不上传 GitHub。
- `docs`：协议、来源、偏差和实验资产保存约定。

## 服务器使用

```bash
source /root/autodl-tmp/hammer_setup/activate.sh
```

服务器项目路径为 `/root/autodl-tmp/Hammer`；运行记录路径为 `/root/autodl-tmp/Hammer/research_outputs/hammer15b_baseline`。

## 复现边界

当前版本组合是兼容性工程选择，并非作者提供的锁定环境。未经实际指标比对，不将环境检查或推理跑通写为 reproduced。公开权重评测与重新训练分别记录。

模型和数据仅保存来源、revision 和校验清单，不上传到 GitHub。任何训练 checkpoint 都保留在服务器，并记录完整路径与校验值。

## 本次结果

| 数据集 | F1_api % | F1_param % |
|---|---:|---:|
| apibank_l1 | 83.38 | 73.90 |
| apibank_l2 | 77.88 | 60.50 |
| NexusRaven | 80.46 | 59.34 |
| sealtool | 95.06 | 89.46 |
| toolalpaca | 80.75 | 53.39 |

状态为 partially_reproduced：这是官方 checkpoint 的五集评测，尚未重训、执行 BFCL 或消融。NexusRaven 参数 F1 比论文参考高 2.46pp，超出预先记录的 2pp 容差；ToolAlpaca 样本数与论文文字存在差异。详细解释见 docs/reproduction_report.md。

requirements.lock.txt 中 torch 的本地 wheel URL 已归一化为 torch==2.5.1；原始 pip freeze 和完整运行记录保留在服务器及本地备份。
