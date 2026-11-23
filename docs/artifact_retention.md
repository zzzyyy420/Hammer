# 实验资产保存约定

## 每次运行

保存 run ID、开始结束时间、源码 commit、环境 lockfile、完整有效配置、命令、随机种子、模型与数据 revision、硬件信息、数据划分与 SHA-256 清单。

保存原始 stdout/stderr、错误日志、GPU 显存、耗时、逐条预测、未经改写的标签来源与汇总指标；失败和负面结果不得覆盖。评测不用于事后更改训练标签或挑选测试最优模型。

## Checkpoint

- 官方 checkpoint：原始权重、config、tokenizer、generation config 和校验值保持不变。
- 训练 checkpoint：保留中间和最终 LoRA adapter，以及 optimizer、scheduler、RNG、trainer state 和断点续训所需信息。
- 不设置自动淘汰 checkpoint；磁盘不足先停止并报告，不静默删除。
- 最终 merged model 单独保存，与 adapter 和原始基座分开。只有实测产物才登记为已生成。

## 当前状态

官方 Hammer-1.5B checkpoint 已存在；训练尚未启动，因此没有新的训练 checkpoint。训练日志目前是状态记录，不包含 loss 或优化步。

## GitHub 与服务器

GitHub 保存代码、配置、安装/运行脚本、简要进度和指标。模型、大数据集、缓存、逐条敏感输出和完整 checkpoint 留在服务器；GitHub 只保存路径和校验摘要。账号凭据、密码、token 不写入仓库。
