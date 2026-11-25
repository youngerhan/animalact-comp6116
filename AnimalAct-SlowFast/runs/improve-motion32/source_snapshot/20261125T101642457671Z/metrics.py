"""Clip-level multi-label metrics using unthresholded scores for AP."""

import numpy as np

THRESHOLDS = np.arange(5, 100, 5) / 100


def average_precision(target, score):
    positives = int(target.sum())
    if positives == 0:
        return None
    order = np.argsort(-score, kind="stable")
    sorted_score, sorted_target = score[order], target[order]
    last_at_threshold = np.r_[np.flatnonzero(np.diff(sorted_score)), len(order) - 1]
    true_positives = np.cumsum(sorted_target)[last_at_threshold]
    precision = true_positives / (last_at_threshold + 1)
    return float(np.dot(np.diff(np.r_[0, true_positives]) / positives, precision))


def confusion_counts(targets, scores, threshold):
    predictions = scores >= threshold
    return (
        (targets & predictions).sum(axis=0),
        (~targets & predictions).sum(axis=0),
        (targets & ~predictions).sum(axis=0),
        (~targets & ~predictions).sum(axis=0),
    )


def f1_scores(tp, fp, fn):
    denominator = 2 * tp + fp + fn
    return np.divide(2 * tp, denominator, out=np.zeros(len(tp)), where=denominator != 0)


def select_threshold(targets, scores):
    def selection_key(threshold):
        counts = confusion_counts(targets, scores, threshold)
        return (f1_scores(*counts[:3]).mean(), -abs(threshold - 0.5))
    return float(max(THRESHOLDS, key=selection_key))


def evaluate(targets, scores, classes, threshold):
    if targets.ndim != 2 or targets.shape != scores.shape or not len(targets) or targets.shape[1] != len(classes):
        raise ValueError("Expected aligned, non-empty targets and scores.")
    if not np.isfinite(scores).all():
        raise ValueError("Model emitted non-finite scores.")
    if targets.dtype != np.bool_:
        raise ValueError("Targets must be boolean indicators.")
    aps = [average_precision(targets[:, col], scores[:, col]) for col in range(len(classes))]
    tp, fp, fn, tn = confusion_counts(targets, scores, threshold)
    per_class_f1 = f1_scores(tp, fp, fn)
    supported = [ap for ap in aps if ap is not None]
    return {
        "clips": len(targets),
        "mAP": float(np.mean(supported)) if supported else None,
        "ap_evaluable_classes": len(supported),
        "macro_F1": float(per_class_f1.mean()),
        "threshold": threshold,
        "per_class": [
            {**item, "AP": aps[col], "F1": float(per_class_f1[col]),
             "positives": int(tp[col] + fn[col]),
             "TP": int(tp[col]), "FP": int(fp[col]), "FN": int(fn[col]), "TN": int(tn[col])}
            for col, item in enumerate(classes)
        ],
    }



def per_class_thresholds(targets, scores):
    chosen = []
    for column in range(scores.shape[1]):
        best, best_f1 = THRESHOLDS[0], -1.0
        for threshold in THRESHOLDS:
            prediction = scores[:, column] >= threshold
            tp = float((targets[:, column] & prediction).sum())
            fp = float((~targets[:, column] & prediction).sum())
            fn = float((targets[:, column] & ~prediction).sum())
            value = float(f1_scores(np.array([tp]), np.array([fp]), np.array([fn]))[0])
            if value > best_f1:
                best, best_f1 = float(threshold), value
        chosen.append(best)
    return chosen


def macro_f1_at(targets, scores, thresholds):
    values = []
    for column, threshold in enumerate(thresholds):
        prediction = scores[:, column] >= threshold
        tp = float((targets[:, column] & prediction).sum())
        fp = float((~targets[:, column] & prediction).sum())
        fn = float((targets[:, column] & ~prediction).sum())
        denominator = 2 * tp + fp + fn
        values.append(2 * tp / denominator if denominator else 0.0)
    return float(np.mean(values))


def summarise(report, targets, scores, thresholds):
    """Add the per-class-threshold view and a class-support cut to a report."""
    report["macro_F1_per_class_threshold"] = macro_f1_at(targets, scores, thresholds)
    report["per_class_thresholds"] = thresholds
    per_class = [(item["AP"], item["positives"]) for item in report["per_class"]]
    for minimum in (10, 20):
        subset = [ap for ap, positives in per_class if ap is not None and positives >= minimum]
        report[f"mAP_positives_ge_{minimum}"] = float(np.mean(subset)) if subset else None
    return report


