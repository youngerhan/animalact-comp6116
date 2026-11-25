# 2026-11-25 本地改进实验

复用现有 Windows / Python 3.11 / torch 2.7.1+cu128 / RTX 5060 Ti 8 GB 环境，不安装新依赖。原模型和既有测试预测保留。

## 事先确定的实验

从 pilot32-whole224/best.pth 开始，32 帧、224 px、全骨干微调、平方根类别重采样。每个变体最多 4 轮，无验证 mAP 改善连续 2 轮停止，初始权重作为第 0 轮参与选优。分类头学习率 1e-4、骨干 1e-5、batch 8、worker 6、seed 42。

1. context32：仅训练时把原始输入区间至少扩至 0.75 秒，作为增强；验证和测试严格限制在原片段内，原目标保持不变。
2. motion32：用帧间差分自动建议区域，训练时 70% 概率用区域、其余整帧；推理用区域。不使用人工框，它不是动物检测器，静止或不可靠区域回退整帧。
3. asymmetric32：使用非对称损失；gamma_negative=4、gamma_positive=0、clip=0.05，按标签判断数取均值。参见 Ridnik et al., ICCV 2021: https://openaccess.thecvf.com/content/ICCV2021/html/Ridnik_Asymmetric_Loss_for_Multi-Label_Classification_ICCV_2021_paper.html

## 选择与评估

每个检查点按验证 mAP 选优。比较普通输入与水平翻转平均，按验证 mAP 和 Macro-F1 均值选择（mAP 相对普通输入最多降低 0.1 个百分点）。融合只在验证集上搜索 0.25 步长，保留至少 25% 原融合。候选必须不降低原验证 mAP 和 Macro-F1，再按二者均值选取。

把候选、权重、阈值写入 selection-frozen.json 后，才计算被选新模型的测试预测；不根据测试结果重选模型。报告同时给出误报、漏报以及按视频配对重采样的不确定性。

单一随机种子；这是有限实验搜索，不承诺收益或上限。时间上下文可能包含其他动作，运动区域可能跟随背景或其他动物，因此均需验证，不能预先认定有效。所有标注和官方划分不变。

## 运行

在项目目录用现有 Python 运行：`python -u improve.py`。进度在 progress.json，各实验日志与最终 summary.json 位于本目录。新权重放在 runs/improve-*，最终融合放在 runs/improved-final-20261125。

