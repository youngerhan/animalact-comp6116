# AnimalAct 动物行为识别

COMP6116 Stage 2：用 AnimalAct 官方标注微调 Kinetics-400 预训练 SlowFast R50，识别 31 类行为，可同时输出多个动作。WildVision 人工标注是另一项考核。

## 目录和入口

| 文件 | 用途 |
|---|---|
| `clips.py` | 按官方划分生成 31 类片段清单 |
| `train.py` | 初版固定间隔采样基线，保留用于对照 |
| `train_strong.py` | 主训练入口：均匀采样、整帧/裁剪、重采样、全骨干微调 |
| `evaluate_run.py` | 验证集选择测试时预处理，再生成配套的验证和测试报告 |
| `predict.py` | 单个检查点的视频预测，默认使用 32 帧短训模型 |
| `ensemble.py` / `blend.py` | 等权集成 / 在验证集选择两组预测的融合比例 |
| `runtime.py` / `metrics.py` | 公共模型执行、保存和指标实现 |
| `predictions.py` / `provenance.py` | 预测顺序校验 / 输入与源码指纹、环境和源码快照 |
| `reproduce.py` / `recipes/` | 打印或执行明确的训练参数 |
| `verify.py` / `test_pipeline.py` | 数据、指标和关键行为检查 |
| `report_results.py` / `report_final.py` | 基线与改进结果图表 |
| `clips_all.py` / `video_level.py` | 可选 119 类实验 / 已覆盖视频的探索性聚合 |

`data/` 是生成的清单；`runs/` 是原始实验、权重和预测；`results/` 是图表与报告。旧的两套重训脚本、物种先验试验和一次性 pilot 收尾脚本已移出工作目录，原文件在项目上一级的清理备份 ZIP 中。

## 环境

已验证的本机环境：Windows、Python 3.11、RTX 5060 Ti 8 GB、PyTorch 2.7.1+cu128、MMAction2 1.2.0、MMCV-lite 2.1.0、MMEngine 0.10.7。完整安装版本在 `requirements-lock.txt`，无需为本次清理重新安装。

在本目录打开 PowerShell。现有环境：

```powershell
$python = 'D:\AnimalAct-setup\envs\animalact\python.exe'
$env:PATH = 'D:\AnimalAct-setup\envs\animalact;D:\AnimalAct-setup\envs\animalact\Library\bin;D:\AnimalAct-setup\envs\animalact\Scripts;' + $env:PATH
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONDONTWRITEBYTECODE = '1'
```

换电脑时创建 Python 3.11 环境，在本目录执行：

```powershell
python -m pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-lock.txt
python -m pip install --no-deps -e ../mmaction2-main
```

数据 `../AnimalAct`、源码 `../mmaction2-main`、预训练权重 `../slowfast_r50_8xb8-4x16x1-256e_kinetics400-rgb_20220901-701b0f6f.pth` 应与项目同级。可用 `--dataset`、`--config`、`--weights` 指定其他位置。依赖锁记录当前环境，未在另一台电脑全新安装验证。

## 验证与训练

```powershell
& $python -m unittest -v test_pipeline
& $python verify.py
& $python clips.py --output data/main-rebuilt.json
# 先查看命令，再加 --execute 开始训练；输出目录必须不存在。
& $python reproduce.py recipes/whole224-clean.json --out runs/whole224-clean-repeat
& $python reproduce.py recipes/whole224-clean.json --out runs/whole224-clean-repeat --execute
# 原训练命令追加 --resume，可从最后一个完整 epoch 恢复。
```

`whole224-clean.json` 从 Kinetics 权重训练 16 帧、224 px 整帧模型，优先使用确定性运算。CUDA 的 max_pool3d 反向算子没有确定性实现，`--deterministic` 会对此发出警告并继续训练。新运行自动保存命令、包版本、输入 SHA256、源码 SHA256 和源码副本。固定种子控制随机性，但该算子及不同 GPU、驱动和库版本会带来数值差异，不承诺逐位一致。

`recipes/historical/` 保存从旧 `settings.json` 导出的参数，可用同一入口复用。包含 `initial` 的实验依赖对应前序权重。旧实验仍使用旧路径记录；导出配方改为项目相对路径。**早期随机裁剪实现有过修复，历史配方只恢复参数，不承诺用当前代码重现旧训练数值**；需要审计原实现时查阅清理备份。原报告数值由已有预测文件复算，不需要重新训练。

## 评估与看结果

```powershell
# 新训练目录：验证集选择 TTA，并同步保存验证/测试分数及指标。
& $python evaluate_run.py --run runs/whole224-clean-repeat
# 使用单模型训练时的验证预处理；该命令不执行多模型集成或选定的 TTA。
& $python predict.py '../AnimalAct/videos/valid/FLBGMVYU.mp4' --start-frame 22 --end-frame 96
# 从已保存预测复算历史融合，使用新的输出目录。
& $python blend.py --base runs/ensemble7 --new runs/pilot32-whole224 --out runs/blend-reproduced --allow-legacy
```

新集成默认核对类别编号、视频编号和起止帧，拒绝错位。历史部分验证预测缺少身份字段，只有显式使用 `--allow-legacy` 才允许融合；此时只能核对已有字段和标签，不能完全证明同标签片段的顺序。

| 已保存实验（946 测试片段） | mAP | Macro-F1 |
|---|---:|---:|
| baseline-stride2 | 40.86% | 36.24% |
| ensemble7 | 48.14% | 46.21% |
| pilot32-whole224 | 46.34% | 44.71% |
| pilot32-blend | 48.38% | 46.67% |

主结果是**已标注动作区间的片段级分类**，不是整段未知视频的自动动作定位。统一阈值、检查点、TTA 和融合比例都从验证集选择。mAP 使用连续分数；Macro-F1 使用验证阈值。每类独立计算 TP/FP/FN/TN；31 类等权平均。稀有类表现较差是本次观察，不能推导理论上限。

`video_level.py` 只聚合有主类片段的 424 个测试视频，遗漏 23 个官方测试视频，不能作为完整 447 视频官方评测。原 `results/` 中的图表和演示属于各自历史实验，最新融合没有独立的实时演示界面。

## 2026-11-25 新一轮改进

最终报告：`results/improvement-20261125/FINAL_REPORT.md`。新模型为 SlowFast + Video Swin-T 混合融合，测试 mAP 49.05%，正则化逐类阈值 Macro-F1 46.67%。mAP 仅小幅提高，Macro-F1 基本持平；未校准统一阈值的 Macro-F1 为 46.41%。本轮不是大幅提升。

实际使用最终十模型预测一个已知区间（末帧不包含在内）：

```powershell
& $python predict_improved.py '../AnimalAct/videos/test/AAWJEBGU.mp4' --start-frame 0 --end-frame 35 --output results/my-prediction.json
# 核验时使用原评估批量大小；重复同一输入以匹配混合精度的运算形状。
& $python predict_improved.py '../AnimalAct/videos/test/AAWJEBGU.mp4' --start-frame 0 --end-frame 35 --match-evaluation-batch --output results/my-prediction-matched.json
# 配方概览、结果复算和回归检查：
& $python predict_improved.py '../AnimalAct/videos/test/AAWJEBGU.mp4' --dry-run
& $python report_improved.py
& $python -m unittest -v test_pipeline test_improvements test_fusion_calibration
```

输出包含每类分数、判断阈值、预测动作和每个成员模型的分数。这个示例的真值为 Flying、Moving，当前预测只检出 Moving，Flying 仍漏检，不能用作“所有动作均已正确识别”的演示。更换为自己的视频时需指定待识别区间；未指定区间时把全视频作为一个片段，尚不进行自动动作定位。

新增文件：`video_ops.py`（训练前后文和运动区域）、`losses.py`（ASL）、`improve.py`（三个 SlowFast 对照）、`swin_experiment.py`（Swin 训练）、`fusion_calibration.py` / `fuse_swin.py`（验证集融合）、`calibrate_thresholds.py`（验证集阈值校准）、`predict_improved.py`（最终实际推理）、`report_improved.py`（最终报告）、`verify_legacy_inference.py`（旧裁剪兼容核验）。

收到发布包后不要直接对历史实验目录运行 --resume 或训练入口。发布包省略 last.pth，不能恢复原优化器状态。要重训，先在独立项目副本中使用新输出目录并保留所需起始权重；不要覆盖现有 `runs/`。本轮执行顺序和关键命令如下：

```powershell
& $python improve.py --stage all
& $python swin_experiment.py
# 仅用于自己刚完成前3轮、且输出目录中存在 last.pth 的新训练；发布包不含 last.pth。
# 满足 SWIN_CONTINUATION_RULE.md 的预先约定条件后才延长：
& $python swin_experiment.py --resume --extra-epochs 3
& $python fuse_swin.py
& $python calibrate_thresholds.py --project . --run runs/improved-swin-final-20261125 --out results/calibration-repeat
& $python report_improved.py
```

Swin 预训练文件 `../swin3d_t-7615ae03.pth` 来自 torchvision 官方 Kinetics-400 权重。新实验的完整参数、命令、源文件快照和输入指纹分别在 `runs/improve-*/` 内；方案在 `results/improvement-20261125/*PLAN.md`，实际记录在同目录日志。重新训练不保证逐位一致；本机 checkpoint 推理核验与重新训练复现是不同层次的验证。

旧 `ft256` 历史中心裁剪使用向下取整，兼容方式单独记录在 `runs/ft256/inference_overrides.json`。默认训练和其他模型保留现有四舍五入方式。`legacy-inference-verification.json` 记录其全部验证/测试预测核验。

本轮通过 GitHub v1.1.0-20261125 增量发布包提供；先解压 v1.0.0 完整包，再应用增量。旧版保留用于对照。

