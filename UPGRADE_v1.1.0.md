# 本轮更新：v1.1.0-20261125

完整复现需先取得 [v1.0.0 的完整离线包](https://github.com/youngerhan/animalact-comp6116/releases/tag/v1.0.0-20261125)，再使用 [v1.1.0 增量包](https://github.com/youngerhan/animalact-comp6116/releases/tag/v1.1.0-20261125)。新包包含代码、新模型最佳检查点、Swin 预训练权重、原始分数和完整报告。旧包提供视频、旧模型、Python 运行时和离线依赖；新增模型沿用已有 torchvision 0.22.1，无需新增依赖。

更新后运行 VERIFY_IMPROVED.cmd 复算新结果；PREDICT_IMPROVED.cmd 执行实际十模型预测。CHECK_FILES.cmd 核验更新后的全包文件。

| 测试片段级指标 | 旧融合 | 新融合＋校准 |
|---|---:|---:|
| mAP |48.38%|49.05%|
| Macro-F1 |46.67%|46.67%|
| Precision（micro） |50.53%|52.11%|
| Recall（micro） |70.00%|70.25%|

新方案统一阈值的 Macro-F1 为 46.41%，逐类校准后为 46.67%。提升较小，未证明显著提升。最终10模型推理比旧8模型成本更高。

原始训练记录、冻结计划和源码指纹保留。便携包仅适配运行路径、历史裁剪及发布说明，packaging/original-settings-v1.1.0/ 保留原设置；训练来源指纹与发布文件校验为不同记录。

为了控制体积，不含优化器 last.pth 和 Swin 中间特征缓存。可用 best.pth 推理与初始化新训练；恢复原优化器和学习率状态的中途续训需要原本地 last.pth。重新训练不保证逐位一致，已有检查点推理和已保存指标复算均可核验。
