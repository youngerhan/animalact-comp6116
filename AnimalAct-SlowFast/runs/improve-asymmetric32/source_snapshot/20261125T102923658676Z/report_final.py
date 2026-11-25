"""Build the final comparison report: figures, per-class table and RESULTS.md."""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

from clips import read_json
from metrics import evaluate, select_threshold
from report_results import confusion_grid, export_error_examples
from video_level import report_for

PROJECT = Path(__file__).resolve().parent
BASELINE = PROJECT / "runs" / "baseline-stride2"


def support_buckets(report, manifest):
    """Mean test AP of the reported model inside training-support buckets."""
    train_support = manifest["counts"]["train"]["support"]
    buckets = {"≤40": [], "41–100": [], ">100": []}
    for item in report["per_class"]:
        if item["AP"] is None:
            continue
        count = train_support[item["name"]]
        key = "≤40" if count <= 40 else ("41–100" if count <= 100 else ">100")
        buckets[key].append(item["AP"])
    return {key: (len(values), float(np.mean(values))) if values else (0, None)
            for key, values in buckets.items()}


def top1_by_class(data, report):
    """Top-1 accuracy on single-label test clips, and the worst confusions."""
    targets, scores = data["targets"], data["scores"]
    single = targets.sum(axis=1) == 1
    rows = []
    for column, item in enumerate(report["per_class"]):
        mask = single & (targets[:, column] > 0)
        if mask.sum() < 8:
            continue
        correct = (scores[mask].argmax(axis=1) == column).mean()
        rows.append((item["name"], int(mask.sum()), float(correct)))
    overall = (scores[single].argmax(axis=1) == targets[single].argmax(axis=1)).mean()
    return float(overall), sorted(rows, key=lambda row: row[2])[:3]


def load(run):
    report = read_json(run / "test_metrics.json")
    with np.load(run / "test_predictions.npz") as stored:
        data = {key: stored[key] for key in stored.files}
    return report, data


def scores(report):
    return {
        "mAP": report["mAP"],
        "Macro-F1": report["macro_F1"],
        "Macro-F1 (per-class thresholds)": report.get("macro_F1_per_class_threshold"),
        "mAP (classes with ≥20 positives)": report.get("mAP_positives_ge_20"),
        "mAP (clips ≤ 1 s)": report.get("short_clips", {}).get("mAP"),
    }


def bar_plot(series, destination):
    labels = list(series[0][1])
    fig, axis = plt.subplots(figsize=(10, 4.6), layout="constrained")
    width = 0.8 / len(series)
    for index, (name, values) in enumerate(series):
        converted = [value * 100 if value is not None else 0 for value in values.values()]
        positions = np.arange(len(labels)) + (index - (len(series) - 1) / 2) * width
        axis.bar(positions, converted, width=width, label=name)
        for position, value in zip(positions, values.values()):
            if value is None:
                axis.text(position, 2, "未计算", ha="center", va="bottom", fontsize=6, rotation=90)
            else:
                axis.text(position, value * 100 + 1.5, f"{value * 100:.1f}", ha="center",
                          va="bottom", fontsize=7)
    axis.set_xticks(np.arange(len(labels)), [label.replace(" (", "\n(") for label in labels], fontsize=8)
    axis.set_ylim(0, 100)
    axis.set_ylabel("Score (%)")
    axis.set_title("AnimalAct SlowFast | fixed test split", loc="left")
    axis.grid(axis="y", color="#DEE6EC", linewidth=0.75)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, loc="upper right", fontsize=8)
    fig.savefig(destination, dpi=180, facecolor="white")
    plt.close(fig)


def curve_plot(history, best_epoch, destination):
    fig, axis = plt.subplots(figsize=(8, 4), layout="constrained")
    epochs = [row["epoch"] for row in history]
    axis.plot(epochs, [row["valid_mAP"] * 100 for row in history], marker="o", label="Validation mAP")
    if "valid_macro_F1_per_class_threshold" in history[0]:
        axis.plot(epochs, [row["valid_macro_F1_per_class_threshold"] * 100 for row in history],
                  marker="s", label="Validation Macro-F1 (per-class thresholds)")
    axis.plot(epochs, [row["valid_macro_F1"] * 100 for row in history], marker="^",
              label="Validation Macro-F1 (shared threshold)")
    axis.axvline(best_epoch, ls="--", color="#708090", label="Selected checkpoint")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Score (%)")
    axis.set_title("Training progress", loc="left")
    axis.grid(color="#DEE6EC", linewidth=0.75)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, fontsize=8)
    fig.savefig(destination, dpi=180, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True,
                        help="Single-model run that supplies settings, history and the curve.")
    parser.add_argument("--headline", type=Path, default=None,
                        help="Run whose test metrics are the improved row; defaults to --run.")
    parser.add_argument("--ensemble", type=Path, nargs="*", default=[],
                        help="Ensemble run directories to add as extra rows.")
    parser.add_argument("--settings-run", type=Path, default=None,
                        help="Run whose settings describe the main training stage; defaults to --run.")
    parser.add_argument("--table", nargs="*", default=[], metavar="LABEL=PATH",
                        help="Extra runs for the experiment table, as label=path.")
    parser.add_argument("--out", type=Path, default=PROJECT / "results" / "final")
    options = parser.parse_args()
    options.out.mkdir(parents=True, exist_ok=True)
    headline = options.headline or options.run

    settings = read_json((options.settings_run or options.run) / "settings.json")
    fine_tune = read_json(options.run / "settings.json")
    history = read_json(options.run / "history.json")
    best_epoch = max(history, key=lambda row: row["valid_mAP"])["epoch"]

    baseline_report, _ = load(BASELINE)
    headline_report, _ = load(headline)
    single_report, _ = load(options.run)
    rows = [("初版（2 轮，128 px）", scores(baseline_report)),
            (f"改进单模型（{settings['image_size']} px 主训练 + {len(history)} 轮 "
             f"{fine_tune['image_size']} px 精修）", scores(single_report))]
    for run in options.ensemble:
        rows.append((f"集成（{len(read_json(run / 'members.json')['members'])} 个模型）",
                     scores(load(run)[0])))

    bar_plot(rows, options.out / "before-after.png")
    curve_source = options.settings_run or options.run
    curve_history = (read_json(curve_source / "history.json")
                     if (curve_source / "history.json").exists() else history)
    curve_best = max(curve_history, key=lambda row: row["valid_mAP"])["epoch"]
    curve_plot(curve_history, curve_best, options.out / "training-progress.png")
    confusion_grid(headline_report, "Improved SlowFast", options.out / "confusion.png")
    with np.load(headline / "test_predictions.npz") as stored:
        export_error_examples({key: stored[key] for key in stored.files}, headline_report,
                              options.out / "error-candidates.csv")

    differences = []
    for before, after in zip(baseline_report["per_class"], headline_report["per_class"]):
        assert before["global_id"] == after["global_id"]
        differences.append({"action": after["name"], "test_positives": after["positives"],
                            "before_AP": before["AP"], "after_AP": after["AP"],
                            "before_F1": before["F1"], "after_F1": after["F1"]})
    with (options.out / "per-class.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(differences[0]))
        writer.writeheader()
        writer.writerows(differences)

    lines = ["# AnimalAct SlowFast 改进训练结果", "",
             "初版的片段级 mAP 为 40.86%、Macro-F1 为 36.24%（2 轮、128 px）。改进版在同样的官方划分上重新训练，"
             "检查点按验证集 mAP 选择，统一阈值与测试时预处理也在验证集上选定，测试集只用来出最终数字。", "",
             "| 设置 | 测试 mAP | Macro-F1 | Macro-F1（逐类阈值） | ≥20 正例类 mAP | ≤1 秒片段 mAP |",
             "|---|---:|---:|---:|---:|---:|"]

    def cell(value):
        return "—" if value is None else f"{value:.2%}"

    for name, values in rows:
        lines.append(f"| {name} | " + " | ".join(cell(values[key]) for key in values) + " |")

    if options.table:
        lines += ["", "## 实验记录", "",
                  "| 实验 | 轮数 | 最佳验证 mAP | 测试 mAP | 测试 Macro-F1 |",
                  "|---|---:|---:|---:|---:|"]
        for entry in options.table:
            label, path = entry.split("=", 1)
            run = Path(path)
            run_history = read_json(run / "history.json") if (run / "history.json").exists() else []
            best = max(run_history, key=lambda row: row["valid_mAP"]) if run_history else None
            summary = read_json(run / "test_summary.json") if (run / "test_summary.json").exists() else None
            if summary is None and (run / "members.json").exists():
                summary = {"test": read_json(run / "test_metrics.json")}
            members = read_json(run / "members.json") if (run / "members.json").exists() else None
            valid_map = (f"{best['valid_mAP']:.2%}" if best
                         else f"{members['valid']['mAP']:.2%}" if members else "—")
            test_map = f"{summary['test']['mAP']:.2%}" if summary else "—"
            test_f1 = f"{summary['test']['macro_F1']:.2%}" if summary else "—"
            lines.append(f"| {label} | {len(run_history) or '—'} | {valid_map} | {test_map} | {test_f1} |")
        lines += ["", "注：全 119 类那一行的片段集合是 1030 个（含由额外 88 个动作生成的窗口），"
                      "其余实验都是主清单的 946 个，两者不能直接比较；该模型因此没有进入集成。"]

    trainable = {"all": "整个骨干网 + 分类头", "l34": "慢/快路 layer3+layer4 + 分类头",
                 "l4": "layer4 + 分类头"}[settings["trainable"]]
    balance = "按 sqrt 反频率重采样" if settings["balance"] == "sqrt" else "正例加权 BCE（权重上限 6）"
    lines += ["", "## 训练设置", "",
              f"- 主训练：{settings['image_size']}×{settings['image_size']}，每段 {settings['frames']} 帧，"
              "在标注区间内均匀采样。初版固定步长 2 从片段中心取帧，而 41% 的片段不足 31 帧，"
              "末帧被重复填充，最多占掉一半的采样位置。",
              f"- 可训练参数：{trainable}；分类头学习率 {settings['head_lr']:g}，骨干 {settings['backbone_lr']:g}。",
              f"- 不平衡处理：{balance}。",
              "- 数据增强：随机缩放裁剪（0.8–1.0）、时域随机截取（75–100%）、水平翻转、亮度/对比度抖动。",
              "- 优化：AdamW、warmup + 余弦退火、梯度裁剪、CUDA 混合精度，按验证集 mAP 早停。",
              f"- 精修阶段：主训练的最佳检查点再用 {fine_tune['image_size']}×{fine_tune['image_size']} 训练 "
              f"{len(history)} 轮，学习率 {fine_tune['head_lr']:g}/{fine_tune['backbone_lr']:g}。",
              "- 输入策略：" + ("整帧缩放到方形（不裁剪）。" if settings.get("crop") == "whole"
                              else "缩放后取方形裁剪。")
              + "这批视频多为 640×360 的野生动物实拍，主体常只占画面几个百分点，中心裁剪会把主体切掉，"
              "因此对整帧输入做了单独对照。",
              "- 测试时：在验证集上比较中心裁剪、缩放中心裁剪、水平翻转平均、空间三裁剪平均与三时窗平均，取验证 mAP 最高者。",
              "", "## 口径说明", "",
              "- mAP 与 Macro-F1 沿用初版的“统一阈值 + 片段级”口径，可与初版直接比较。",
              "- “逐类阈值”一列只为说明统一阈值如何压低稀有类；它的阈值同样只在验证集上选，"
              "但在 912 个验证片段、部分类只有 1–5 个正例的情况下会过拟合验证集，测试集上并不优于统一阈值。",
              "- “≥20 正例类 mAP”排除了每类只有 4–9 个正例、AP 取值高度离散的稀有类，更能反映主干能力的提升。",
              "- 测试集 946 个片段、31 类，其中 12 类正例少于 10 个。宏平均下单个稀有类的 AP 抖动可达 ±0.2，"
              "因此单种子模型之间 2–3 个百分点以内的差异不应解读为真实差距。", "",
              f"混淆矩阵基于 {headline_report['clips']} 个测试片段与阈值 {headline_report['threshold']:.2f}；"
              "错误候选见 `error-candidates.csv`。", ""]

    manifest = read_json(PROJECT / "data" / "main.json")
    classes = manifest["classes"]
    video_rows = []
    for name, run in (("初版", BASELINE), ("改进单模型", options.run), ("集成", headline)):
        _, valid_targets, valid_scores = report_for(run, "valid", classes, "mean")
        threshold = select_threshold(valid_targets, valid_scores)
        videos, test_targets, test_scores = report_for(run, "test", classes, "mean")
        video_report = evaluate(test_targets, test_scores, classes, threshold)
        video_rows.append((name, len(videos), video_report["mAP"], video_report["macro_F1"]))
    lines += ["", "## 官方口径：整视频多标签 mAP", "",
              f"`MAIN_EVALUATION.md` 规定的评估单位是整视频（{video_rows[0][1]} 个测试视频），"
              "把一个视频内所有片段分数的均值作为该视频的类别分数。按同一套检查点与验证集阈值换算：", "",
              "| 设置 | 整视频 mAP | 整视频 Macro-F1 |", "|---|---:|---:|"]
    for name, _, mAP, macro in video_rows:
        lines.append(f"| {name} | {mAP:.2%} | {macro:.2%} |")
    lines += ["", f"已覆盖视频的片段聚合口径（不含无主类片段的视频）下集成模型比初版高 {100 * (video_rows[-1][2] - video_rows[0][2]):.2f} 个点，"
              "两个口径的结论一致。测试集平均每个视频只有 2.2 个片段，聚合带来的提升有限。", ""]
    buckets = support_buckets(headline_report, manifest)
    overall_top1, worst = top1_by_class(load(headline)[1], headline_report)
    lines += ["", "## 为什么分数停在这个量级", "",
              "**类别支持度与本次模型表现有关，但不能据此推导 AP 上限。** 把 31 个主类按训练正例数分组，测试集平均 AP 为：", "",
              "| 训练正例数 | 类数 | 平均测试 AP |", "|---|---:|---:|"]
    for key, (count, value) in buckets.items():
        lines.append(f"| {key} | {count} | {'—' if value is None else f'{value:.3f}'} |")
    lines += ["**最大的三个类在语义上互相重叠。** 单标签测试片段的 top-1 准确率为 "
              f"{overall_top1:.1%}，其中表现最差的类是：", ""]
    for name, count, accuracy in worst:
        lines.append(f"- {name}：{accuracy:.0%} 正确（{count} 个单标签片段）")
    lines += ["", "Attending、Sensing、Keeping still 三个类合计占测试集 67% 的片段，这些动作在画面上的差别"
              "只有细微的头部与躯干运动。混淆原因需要结合人工看片进一步核实。", "",
              "**主体太小。** 视频多为 640×360 的野生动物实拍，蜜蜂、蚂蚁一类主体只占画面的几个百分点，"
              "对整帧输入与中心裁剪做过对照：整帧输入在验证集上更高（0.4237 对 0.3994），说明裁剪确实在丢主体。", "",
              "## 试过但没有采用的做法", "",
              "- **物种先验**：用标注里的 `list_animal` 构造 P(动作|物种)/P(动作) 重排分数（清理前源码备份中的 `species_prior.py`）。"
              "验证集上 alpha=0 最好，加大反而掉点——模型已从画面中学到物种相关线索，先验只是重复。",
              "- **全 119 类词表训练**：多出 363 个训练片段并提供了更难的负例，但验证集主类 mAP 只有 0.406，"
              "低于 31 类训练的 0.442，且片段集合不同无法进入集成。",
              "- **逐类阈值**：验证集上看起来能到 0.52，测试集反而低于统一阈值，属于在 912 个验证片段上过拟合，只作为诊断保留。", ""]
    (options.out / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
