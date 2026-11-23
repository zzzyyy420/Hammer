# Hammer 1.5B 全量训练进度

状态：训练进行中，尚未产生全量训练结果。

- 输入：60,000 条 xLAM 与 7,500 条 irrelevance 数据，均已校验 JSON、样本数和 SHA-256。
- 按作者预处理扩增到 202,500 条；保留发布数据中的重复项。
- 五个测试集的 1,252 条 query 与原始训练 query 未发现精确归一化重合；不等于无语义或预训练污染。
- 原始基座：Qwen/Qwen2-1.5B-Instruct，revision 2ebb13f1f55edeb5bd4b108d0ad589397a6ee4ac。
- BF16 LoRA rank 32，所有线性层；cutoff 2048；每卡 batch 4，累积 2；lr 5e-5；cosine；1 epoch；seed 42。
- 官方 1.5B checkpoint 五集评测已单独完成；本次是从原始基座重训。
- 2 步训练、checkpoint 保存、从第 2 步恢复到第 3 步均已验证。小步结果不作为训练后指标。
- 可续训 checkpoint 约 1GB，为保留全部产物将全量保存间隔设为 2500 步，保存 optimizer、scheduler、RNG、trainer state；保留最终 adapter 与合并模型。
- 实测开始阶段显存约 21.3GB，初期速度预估余下约 8 小时，非完成承诺。
- 磁盘低于 5GiB 或出现非有限值会停止，保留现场。
- 用户授权训练、落盘和训练后评测结束后自动关机；确认不可恢复故障时也关机。使用 /usr/bin/shutdown，不释放实例或删除数据。普通可修复错误不直接触发关机。

完整日志、数据、模型和 checkpoint 仅在服务器/本地保存，不上传 GitHub。

## 服务器查看

```bash
tail -f /root/autodl-tmp/Hammer/research_outputs/hammer15b_train_v1/logs/train_full.log
cat /root/autodl-tmp/Hammer/research_outputs/hammer15b_train_v1/progress.json
```

## 第二次运行

第一次全量训练在约 130 步出现 CUDA OOM，原始日志保留，未产生最终权重。第二次仅启用 expandable_segments 分配器，保持 BF16、micro batch 4、累积 2、模型和数据不变，从原始基座重新开始；并加上第 100 步保存 callback。已越过原故障点，第 100 步 checkpoint 的 adapter、DeepSpeed 模型与 optimizer 状态、trainer state、RNG 文件和 SHA-256 清单已核验。实际序列化 TrainingArguments 确认常规保存间隔 2500。仍在训练中，不代表全量训练成功。
