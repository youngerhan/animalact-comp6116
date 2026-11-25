# AnimalAct COMP6116

基于 MMAction2 和 Kinetics-400 预训练 SlowFast R50 与 Video Swin-T 的动物行为多标签识别项目。

本仓库用于浏览源码、数据标注、训练配方、指标与诊断结果。完整离线复现包（包括全部视频、最佳模型权重、预训练权重、Windows Python 运行时和离线依赖）位于本仓库 **Releases** 中。

## 最新版本 v1.1.0（2026-11-25）

本轮改进已发布为 [v1.1.0 增量包](https://github.com/youngerhan/animalact-comp6116/releases/tag/v1.1.0-20261125)：包含新增模型权重、源码、训练记录和评估结果。**先取得 v1.0.0 完整离线包，再下载并应用 v1.1.0 增量**。已有旧完整包的成员无需重下视频与离线环境。

解压增量 ZIP，双击 `APPLY_UPDATE.cmd`，输入已解压旧完整包的根目录。应用后运行 `VERIFY_IMPROVED.cmd` 复算最新指标，运行 `PREDICT_IMPROVED.cmd` 看实际十模型输出。源代码 ZIP 仍不含权重、视频和环境。

完整报告：[本轮结果](AnimalAct-SlowFast/results/improvement-20261125/FINAL_REPORT.md)；详细用法：[更新说明](UPGRADE_v1.1.0.md)。

## 获取基础完整项目

1. 打开本仓库 Releases，进入 `v1.0.0-20261125`。
2. 下载全部 7 个 `AnimalAct-Reproducible-20261125.zip.partNNN` 分卷，以及 `release-parts.json`、`merge_release_parts.ps1` 和 `MERGE_PARTS.cmd`，放在同一目录。
3. Windows 下双击 `MERGE_PARTS.cmd`。它会校验每个分卷，再还原并校验完整 ZIP；不需要预先安装 Python。
4. 完整解压还原的 ZIP，阅读 `00_先读我.md`，运行 `INSTALL_WINDOWS.cmd`，然后运行 `VERIFY_RESULTS.cmd`。

**GitHub 的 Source code ZIP 只包含源码，不包含完整数据、模型与离线环境。上面的安装入口适用于 Release 完整包。**

## 已保存结果

| 评估单位：946 个测试片段、31 类 | mAP | Macro-F1 |
|---|---:|---:|
| 初版基线 | 40.86% | 36.24% |
| 32 帧单模型 | 46.34% | 44.71% |
| 旧融合 v1.0 | 48.38% | 46.67% |
| 新融合 v1.1（逐类阈值校准） | **49.05%** | **46.67%** |

旧的 `PREDICT_DEMO.cmd` 使用 32 帧单模型，`VERIFY_RESULTS.cmd` 验证旧结果；新版本用 `PREDICT_IMPROVED.cmd` 和 `VERIFY_IMPROVED.cmd`。旧融合由原7模型集成和32帧模型各占50%；新融合加入训练前后文模型及Swin，共10模型。新方案统一阈值的 Macro-F1 为46.41%，校准后为46.67%。mAP提升有限，尚未证明显著提升。所有结果按946个标注区间片段、31类统计。已有结果可从原始分数复算；跨硬件重训不保证逐位一致。

## 目录

- `AnimalAct-SlowFast/`：项目代码、数据清单、训练配方、评估和错误诊断。
- `mmaction2-main/`：实际使用的 MMAction2 1.2.0 源码与许可证。
- `AnimalAct/`：数据说明、标注和官方划分；视频在 Release 完整包。
- `packaging/`：离线包适配说明、原始配置及安装验证日志。
- `PACKAGING_VALIDATION.json`：独立目录安装、指标复算、预测和小规模训练的检查记录。

完整离线环境适用于 Windows x64。训练、视频预测和从检查点重新评估需要兼容的 NVIDIA GPU/驱动；指标复算不需要 GPU。macOS/Linux 需另配环境，未实机验证。

项目使用 [MMAction2](https://github.com/open-mmlab/mmaction2) 与 [SlowFast](https://openaccess.thecvf.com/content_ICCV_2019/html/Feichtenhofer_SlowFast_Networks_for_Video_Recognition_ICCV_2019_paper.html)。第三方许可证保留于相应目录；原始数据来源和划分说明见 AnimalAct/README.md。
