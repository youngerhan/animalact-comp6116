# AnimalAct COMP6116

基于 MMAction2 和 Kinetics-400 预训练 SlowFast R50 的动物行为多标签识别项目。

本仓库用于浏览源码、数据标注、训练配方、指标与诊断结果。完整离线复现包（包括全部视频、最佳模型权重、预训练权重、Windows Python 运行时和离线依赖）位于本仓库 **Releases** 中。

## 获取完整项目

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
| 最终融合 | 48.38% | 46.67% |

默认视频预测使用 32 帧单模型。最终融合是原 7 模型集成与 32 帧模型各占 50%。已有结果可从原始分数复算；跨硬件重训不保证逐位一致。

## 目录

- `AnimalAct-SlowFast/`：项目代码、数据清单、训练配方、评估和错误诊断。
- `mmaction2-main/`：实际使用的 MMAction2 1.2.0 源码与许可证。
- `AnimalAct/`：数据说明、标注和官方划分；视频在 Release 完整包。
- `packaging/`：离线包适配说明、原始配置及安装验证日志。
- `PACKAGING_VALIDATION.json`：独立目录安装、指标复算、预测和小规模训练的检查记录。

完整离线环境适用于 Windows x64。训练、视频预测和从检查点重新评估需要兼容的 NVIDIA GPU/驱动；指标复算不需要 GPU。macOS/Linux 需另配环境，未实机验证。

项目使用 [MMAction2](https://github.com/open-mmlab/mmaction2) 与 [SlowFast](https://openaccess.thecvf.com/content_ICCV_2019/html/Feichtenhofer_SlowFast_Networks_for_Video_Recognition_ICCV_2019_paper.html)。第三方许可证保留于相应目录；原始数据来源和划分说明见 AnimalAct/README.md。
