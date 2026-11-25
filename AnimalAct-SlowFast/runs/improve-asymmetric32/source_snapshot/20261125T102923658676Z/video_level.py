"""Exploratory aggregation of annotated clips to covered videos.

Excludes videos without main-class clips; this is not the full official video benchmark.
"""

import argparse
from pathlib import Path

import numpy as np

from clips import read_json, write_json
from metrics import evaluate, select_threshold

PROJECT = Path(__file__).resolve().parent


def pool(rows, targets, scores, how):
    videos, index, pooled_targets = [], {}, []
    for position, row in enumerate(rows):
        if row["video_id"] not in index:
            index[row["video_id"]] = len(videos)
            videos.append(row["video_id"])
            pooled_targets.append(targets[position].copy())
        else:
            pooled_targets[index[row["video_id"]]] |= targets[position]
    pooled = np.zeros((len(videos), scores.shape[1]))
    for position, row in enumerate(rows):
        slot = index[row["video_id"]]
        if how == "max":
            pooled[slot] = np.maximum(pooled[slot], scores[position])
        else:
            pooled[slot] += scores[position]
    if how != "max":
        pooled /= np.bincount([index[row["video_id"]] for row in rows])[:, None]
    return videos, np.asarray(pooled_targets), pooled


def report_for(run, split, classes, how):
    manifest = read_json(PROJECT / "data" / "main.json")
    rows = manifest["splits"][split]
    with np.load(run / f"{split}_predictions.npz") as stored:
        targets, scores = stored["targets"], stored["scores"]
    return pool(rows, targets, scores, how)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--pool", choices=("mean", "max"), default="mean")
    parser.add_argument("--out", type=Path, default=PROJECT / "results" / "video-level.json")
    options = parser.parse_args()
    classes = read_json(PROJECT / "data" / "main.json")["classes"]

    table = []
    for run in options.runs:
        videos, valid_targets, valid_scores = report_for(run, "valid", classes, options.pool)
        threshold = select_threshold(valid_targets, valid_scores)
        test_videos, test_targets, test_scores = report_for(run, "test", classes, options.pool)
        report = evaluate(test_targets, test_scores, classes, threshold)
        supported = [item["AP"] for item in report["per_class"]
                     if item["AP"] is not None and item["positives"] >= 10]
        entry = {"run": run.name, "pool": options.pool, "videos": len(test_videos),
                 "mAP": report["mAP"], "macro_F1": report["macro_F1"],
                 "threshold": threshold, "mAP_positives_ge_10": float(np.mean(supported))}
        table.append(entry)
        print(f"{run.name:<28} videos={len(test_videos):<4} mAP={report['mAP']:.4f} "
              f"macro-F1={report['macro_F1']:.4f} mAP(>=10 positives)={entry['mAP_positives_ge_10']:.4f}",
              flush=True)
    write_json(options.out, table)


if __name__ == "__main__":
    main()
