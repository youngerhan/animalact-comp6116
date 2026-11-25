# AnimalAct v1.1.0 增量更新

此包把 `v1.0.0-20261125` 完整离线复现包更新到本轮训练后的版本。新增 SlowFast 前后文增强实验、Video Swin-T 权重、融合预测、验证集阈值校准、训练记录和最终报告。原数据集、原始标注和官方划分保持原样。

| 946 个测试片段、31 类 | 旧融合 | 本轮融合＋逐类阈值校准 |
|---|---:|---:|
| mAP | 48.38% | 49.05% |
| Macro-F1 | 46.67% | 46.67% |

mAP 小幅增加，Macro-F1 基本持平；不能称为大幅或显著提升。新模型统一阈值下的 Macro-F1 为 46.41%，表中校准使用验证集选择的阈值。最终推理包含 10 个模型，计算成本比原方案更高。

## 首次获取完整项目

1. 在私人仓库的 `v1.0.0-20261125` Release 下载全部 7 个分卷、`release-parts.json`、`merge_release_parts.ps1` 和 `MERGE_PARTS.cmd`。
2. 把这些文件放在同一目录，运行 `MERGE_PARTS.cmd`，完整解压生成的 ZIP。解压后的项目根目录应有 `runtime/`、`AnimalAct/`、`AnimalAct-SlowFast/` 和 `INSTALL_WINDOWS.cmd`。
3. 运行 `INSTALL_WINDOWS.cmd` 安装包内离线依赖。
4. 再按下面的步骤安装本轮增量更新。

GitHub 自动提供的 **Source code ZIP 不含视频、模型权重和离线环境**，不能替代 `v1.0.0` Release 的完整包。已有完整包的组员可以直接安装增量更新，不必重新下载视频和旧权重。

## 安装增量更新

1. 下载本轮增量 ZIP 及其 SHA256 校验文件，将 ZIP 完整解压到旧完整包之外的另一个目录。
2. 双击解压目录中的 `APPLY_UPDATE.cmd`，按提示输入旧完整包的根目录。
3. 程序先核对增量文件、旧权重及数据文件，再更新项目。若目标文件有组员自行修改的内容，会停止并保留该文件；请按提示处理后再运行。
4. 完成后，到旧完整包根目录运行 `VERIFY_IMPROVED.cmd`，复算本轮保存的指标并执行检查。
5. 需要查看真实视频预测时，运行 `PREDICT_IMPROVED.cmd`。它需要兼容的 NVIDIA GPU/驱动，读取已选定的十模型与阈值。
6. `CHECK_FILES.cmd` 可以按更新后的完整 SHA256 清单检查所有归档文件。

被替换的旧文件保存在 `packaging/update-backups/v1.1.0-20261125/`。更新程序不会删除旧实验，也不会启动重训；重新运行时会跳过已安装且校验一致的文件。

命令行示例：

```powershell
# 预检，不写文件：
python apply_update.py --target 'D:\AnimalAct-Reproducible-20261125' --dry-run
# 应用更新：
python apply_update.py --target 'D:\AnimalAct-Reproducible-20261125'
# 再次核验所有增量文件已安装：
python apply_update.py --target 'D:\AnimalAct-Reproducible-20261125' --verify-only
```

没有单独安装 Python 时，`APPLY_UPDATE.cmd` 可以使用旧完整包内的 `runtime/python.exe`，不需要额外下载软件。应用器只使用 Python 标准库。

## 验证范围

保存的连续分数和固定阈值可以复算指标。本轮已经核验实际十模型推理，并单独核对历史 `ft256` 中心裁剪兼容方式。完整结论、局限和逐类结果见：

- `AnimalAct-SlowFast/results/improvement-20261125/FINAL_REPORT.md`
- `AnimalAct-SlowFast/results/improvement-20261125/summary-final.json`
- `AnimalAct-SlowFast/results/improvement-20261125/per-class-final.csv`

这些指标针对已标注区间的多标签分类，尚不包含未知长视频的自动动作定位。已有检查点的推理核验不等于重新训练逐位复现；不同 GPU、驱动和非确定性运算会带来数值差异。本轮增量包不包含优化器恢复文件 `last.pth` 和特征缓存，保留最佳权重与重新执行训练所需的代码、参数和起始权重。
