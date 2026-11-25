"""Verify saved predictions and produce figures and a compact experiment summary."""

import argparse
import csv
from pathlib import Path
from textwrap import fill

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from clips import read_json
from metrics import evaluate

RUNS = {
    "baseline-stride2": "Stride 2 baseline",
    "comparison-stride1": "Stride 1 comparison",
}
COLORS = ("#185C8A", "#C65A22")


def confusion_grid(report, title, destination):
    fig, axes = plt.subplots(7, 5, figsize=(12, 15), layout="constrained")
    for axis, item in zip(axes.flat, report["per_class"]):
        matrix = np.array([[item["TN"], item["FP"]], [item["FN"], item["TP"]]])
        row_sums = matrix.sum(axis=1, keepdims=True)
        normalized = np.divide(matrix, row_sums, out=np.zeros((2, 2)), where=row_sums != 0)
        axis.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
        for row, col in np.ndindex(2, 2):
            axis.text(col, row, f"{matrix[row, col]:,}", ha="center", va="center",
                      color="white" if normalized[row, col] > 0.6 else "#14202B", fontsize=9)
        axis.set_title(fill(item["name"], 22), fontsize=9, pad=4)
        axis.set_xticks([0, 1], ["Absent", "Present"], fontsize=7)
        axis.set_yticks([0, 1], ["Absent", "Present"], fontsize=7)
        axis.tick_params(length=0)
        for spine in axis.spines.values():
            spine.set_visible(False)
    for axis in list(axes.flat)[len(report["per_class"]):]:
        axis.set_visible(False)
    fig.suptitle(f"{title} | test set per-class confusion", fontsize=15)
    fig.supxlabel("Predicted label  |  Cell text shows counts, color shows fraction of the true-label row")
    fig.supylabel("True label")
    fig.savefig(destination, dpi=165, facecolor="white")
    plt.close(fig)


def comparison_plot(reports, destination):
    fig, axis = plt.subplots(figsize=(8, 4.3), layout="constrained")
    x = np.arange(3)
    for index, (name, report) in enumerate(reports.items()):
        values = np.array([report["mAP"], report["macro_F1"], report["short_clips"]["mAP"]]) * 100
        bars = axis.bar(x + (index - 0.5) * 0.30, values, width=0.30,
                        color=COLORS[index], label=RUNS[name])
        axis.bar_label(bars, fmt="%.1f", fontsize=9, padding=3)
    axis.set_xticks(x, ["mAP", "Macro-F1", "mAP on clips ≤ 1 s"])
    axis.set_ylim(0, 100)
    axis.set_ylabel("Score (%)")
    axis.set_title("AnimalAct SlowFast performance on the fixed test split", loc="left")
    axis.grid(axis="y", color="#DEE6EC", linewidth=0.75)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, loc="upper right")
    fig.savefig(destination, dpi=180, facecolor="white")
    plt.close(fig)


def export_error_examples(data, report, destination):
    predictions, targets = data["scores"], data["targets"]
    false_positives = np.argwhere((predictions >= report["threshold"]) & ~targets)
    false_negatives = np.argwhere((predictions < report["threshold"]) & targets)
    rows = []
    for name, array, reverse in (
        ("false_positive", false_positives, True), ("false_negative", false_negatives, False)
    ):
        for clip, class_id in sorted(array, key=lambda item: predictions[tuple(item)], reverse=reverse)[:12]:
            true_labels = [report["per_class"][label]["name"] for label in np.flatnonzero(targets[clip])]
            rows.append({
                "error": name,
                "video": str(Path("AnimalAct") / "videos" / "test" / f"{data['video_ids'][clip]}.mp4"),
                "start_frame": int(data["start_frames"][clip]),
                "end_frame_exclusive": int(data["end_frames"][clip]),
                "class": report["per_class"][class_id]["name"],
                "score": float(predictions[clip, class_id]),
                "annotated_actions": " | ".join(true_labels),
            })
    with destination.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(project):
    manifest = read_json(project / "data" / "main.json")
    destination = project / "results"
    destination.mkdir(exist_ok=True)
    reports, best_epochs = {}, {}
    reference_settings = None
    epochs = None
    for name, title in RUNS.items():
        run = project / "runs" / name
        settings = read_json(run / "settings.json")
        assert settings.pop("stride") == (2 if name == "baseline-stride2" else 1)
        settings.pop("out")
        settings.pop("log_every")
        if reference_settings is None:
            reference_settings = settings
        else:
            assert settings == reference_settings, "Comparison settings differ beyond the sampling stride."
        valid = read_json(run / "valid_metrics.json")
        report = read_json(run / "test_metrics.json")
        history = read_json(run / "history.json")
        if epochs is None:
            epochs = len(history)
        assert len(history) == epochs and epochs > 0, "Compared runs need the same number of training epochs."
        best_epochs[name] = max(history, key=lambda item: item["valid_mAP"])["epoch"]
        assert valid["threshold"] == report["threshold"]
        with np.load(run / "test_predictions.npz", allow_pickle=False) as predictions:
            assert len(predictions["scores"]) == manifest["counts"]["test"]["clips"]
            assert predictions["global_class_ids"].tolist() == [item["global_id"] for item in manifest["classes"]]
            observed_keys = list(zip(predictions["video_ids"], predictions["start_frames"], predictions["end_frames"]))
            manifest_keys = [(r["video_id"], r["start_frame"], r["end_frame"])
                             for r in manifest["splits"]["test"]]
            assert observed_keys == manifest_keys
            wanted = np.zeros_like(predictions["targets"])
            for row_idx, row in enumerate(manifest["splits"]["test"]):
                wanted[row_idx, row["labels"]] = True
            assert np.array_equal(predictions["targets"], wanted)
            checked = evaluate(wanted, predictions["scores"], manifest["classes"], report["threshold"])
            assert checked == {key: report[key] for key in checked}
            short = np.array([row["duration_seconds"] <= 1 for row in manifest["splits"]["test"]])
            checked_short = evaluate(wanted[short], predictions["scores"][short], manifest["classes"],
                                     report["threshold"])
            assert checked_short == {key: report["short_clips"][key] for key in checked_short}
            if name == "baseline-stride2":
                export_error_examples(predictions, report, destination / "baseline_error_candidates.csv")
        assert report["ap_evaluable_classes"] == 31
        confusion_grid(report, title, destination / f"{name}-confusion.png")
        reports[name] = report
    comparison_plot(reports, destination / "test-comparison.png")

    baseline, comparison = reports.values()
    delta = [
        (comparison[key] - baseline[key]) * 100 for key in ("mAP", "macro_F1")
    ] + [(comparison["short_clips"]["mAP"] - baseline["short_clips"]["mAP"]) * 100]
    lines = [
        "# AnimalAct SlowFast 实验结果", "",
        f"31 类动作片段识别，使用官方固定划分。两个模型各使用 seed {reference_settings['seed']} 训练 {epochs} 轮，"
        "以验证集 mAP 选检查点，并使用该检查点在验证集选定的统一阈值计算测试 Macro-F1。", "",
        "| 设置 | 最佳轮次 | 阈值 | 测试 mAP | 测试 Macro-F1 | ≤1 秒片段 mAP |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, report in reports.items():
        lines.append(f"| {RUNS[name]} | {best_epochs[name]} | {report['threshold']:.2f} | "
                     f"{report['mAP']:.2%} | {report['macro_F1']:.2%} | {report['short_clips']['mAP']:.2%} |")
    lines += [
        "", f"Stride 1 相对 Stride 2 的变化依次为 {delta[0]:+.2f}、{delta[1]:+.2f}、{delta[2]:+.2f} 个百分点。",
        f"测试集有 {baseline['clips']} 个片段，31 类均有正例。≤1 秒的子集有 {baseline['short_clips']['clips']} 个片段，"
        f"短时 mAP 在其中有正例的 {baseline['short_clips']['ap_evaluable_classes']} 类上计算。",
        "固定 16 帧时，改变取帧间隔也改变了时间覆盖范围。结果来自单种子对照。",
        "", "训练配置、检查点、原始预测和逐类指标位于各自的 runs 子目录。"
        "错误样本 CSV 按预测分数列出候选，供逐段看片解释。",
    ]
    (destination / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parent)
    main(parser.parse_args().project)

