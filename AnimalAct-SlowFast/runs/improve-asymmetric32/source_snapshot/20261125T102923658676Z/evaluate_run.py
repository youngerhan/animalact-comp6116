"""Choose test-time preprocessing on the validation split, then score the test split once.

Compares three test-time settings of the same checkpoint (plain centre crop, the
ImageNet-style zoomed centre crop, and zoom + horizontal-flip averaging), keeps the
setting with the best validation mAP, and writes the official test report for it.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from clips import read_json, write_json
from metrics import evaluate, select_threshold
from runtime import PROJECT, predict
from train_strong import ClipDataset, build_model, predict_tta
from runtime import run_report
from metrics import per_class_thresholds, summarise

WINDOWS = ((0.0, 1.0), (0.0, 0.8), (0.2, 0.8))
VARIANTS = (
    {"name": "zoom1.00", "zoom": 1.0, "tta": False, "crops": (0.5,), "windows": ((0.0, 1.0),)},
    {"name": "zoom1.15", "zoom": 1.15, "tta": False, "crops": (0.5,), "windows": ((0.0, 1.0),)},
    {"name": "zoom1.15+flip", "zoom": 1.15, "tta": True, "crops": (0.5,), "windows": ((0.0, 1.0),)},
    {"name": "zoom1.15+flip+3crop", "zoom": 1.15, "tta": True, "crops": (0.0, 0.5, 1.0),
     "windows": ((0.0, 1.0),)},
    {"name": "zoom1.15+flip+windows", "zoom": 1.15, "tta": True, "crops": (0.5,), "windows": WINDOWS},
)


def scores(model, rows, settings, mean, std, variant):
    totals, targets, passes = None, None, 0
    crop_mode = getattr(settings, "crop", "random")
    # Whole-frame inputs have no crop position to vary, so spatial averaging would be a repeat.
    crops = variant["crops"] if crop_mode == "random" else (0.5,)
    for window in variant["windows"]:
        for crop_x in crops:
            loader = DataLoader(
                ClipDataset(settings.dataset, rows, len(settings.classes), settings.frames,
                            settings.image_size, False, eval_zoom=variant["zoom"], window=window,
                            crop=crop_mode, crop_x=crop_x,
                            min_context_seconds=getattr(settings, "min_context_seconds", 0.0),
                            spatial_view=getattr(settings, "spatial_view", "whole"),
                            roi_probability=getattr(settings, "roi_probability", .7)),
                batch_size=settings.batch_size, num_workers=settings.workers, pin_memory=True)
            targets, current = (predict_tta(model, loader, mean, std) if variant["tta"]
                                else predict(model, loader, mean, std))
            totals = current if totals is None else totals + current
            passes += 1
    return targets, totals / passes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--main-only", action="store_true",
                        help="Score only the fixed main classes (needed for full-vocabulary runs).")
    parser.add_argument("--reference", type=Path, help="Optional baseline run for comparison.")
    options = parser.parse_args()

    settings = argparse.Namespace(**read_json(options.run / "settings.json"))
    settings.dataset = Path(settings.dataset)
    settings.config = Path(settings.config)
    settings.weights = Path(settings.weights)
    manifest = read_json(Path(settings.manifest))
    classes = manifest["classes"]
    columns = list(range(len(classes)))
    if options.main_only:
        main_ids = [item["global_id"] for item in read_json(PROJECT / "data" / "main.json")["classes"]]
        index = {item["global_id"]: position for position, item in enumerate(classes)}
        columns = [index[global_id] for global_id in main_ids if global_id in index]
        classes = [classes[position] for position in columns]

    def project(pairs):
        targets, current = pairs
        return targets[:, columns], current[:, columns]

    best = torch.load(options.run / "best.pth", map_location="cpu", weights_only=True)
    if best["classes"] != manifest["classes"]:
        raise ValueError("Checkpoint class mapping differs from manifest.")

    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(settings.seed)
    model, _, _ = build_model(settings, len(manifest["classes"]), settings.weights)
    model.load_state_dict(best["state_dict"])
    mean = torch.tensor([123.675, 116.28, 103.53], device="cuda").view(1, 3, 1, 1, 1)
    std = torch.tensor([58.395, 57.12, 57.375], device="cuda").view(1, 3, 1, 1, 1)

    valid_rows = manifest["splits"]["valid"]
    table, chosen = [], None
    for variant in VARIANTS:
        pairs = project(scores(model, valid_rows, settings, mean, std, variant))
        threshold = select_threshold(*pairs)
        thresholds = per_class_thresholds(*pairs)
        report = evaluate(*pairs, classes, threshold)
        summarise(report, *pairs, thresholds)
        table.append({"variant": variant["name"], "zoom": variant["zoom"],
                      "tta": variant["tta"], "windows": len(variant["windows"]),
                      "crops": len(variant["crops"]), "threshold": threshold,
                      "valid_mAP": report["mAP"], "valid_macro_F1": report["macro_F1"],
                      "valid_macro_F1_per_class_threshold": report["macro_F1_per_class_threshold"],
                      "valid_mAP_positives_ge_20": report["mAP_positives_ge_20"]})
        print({key: (round(value, 4) if isinstance(value, float) else value)
               for key, value in table[-1].items()}, flush=True)
        if chosen is None or report["mAP"] > chosen[0]["valid_mAP"]:
            chosen = (table[-1], pairs, threshold, thresholds)

    pick, pairs, threshold, thresholds = chosen
    print(f"Selected test-time setting {pick['variant']} on validation mAP", flush=True)
    run_report(options.run, "valid", pairs, classes, threshold, thresholds, valid_rows)
    chosen_variant = next(variant for variant in VARIANTS if variant["name"] == pick["variant"])
    test_rows = manifest["splits"]["test"]
    pairs = project(scores(model, test_rows, settings, mean, std, chosen_variant))
    report = run_report(options.run, "test", pairs, classes, threshold, thresholds, test_rows)
    reference = read_json(options.reference / "test_metrics.json") if options.reference else None
    summary = {"run": str(options.run), "epoch": best["epoch"], "selected": pick["variant"],
               "valid_variants": table,
               "test": {key: report[key] for key in
                        ("clips", "mAP", "macro_F1", "macro_F1_per_class_threshold",
                         "mAP_positives_ge_10", "mAP_positives_ge_20", "threshold")},
               "test_short_mAP": report.get("short_clips", {}).get("mAP"),
               "reference": ({"mAP": reference["mAP"], "macro_F1": reference["macro_F1"]}
                             if reference else None)}
    write_json(options.run / "test_summary.json", summary)
    print({key: (round(value, 4) if isinstance(value, float) else value)
           for key, value in summary["test"].items()}, flush=True)
    if reference:
        print(f"delta vs reference: mAP {100 * (summary['test']['mAP'] - reference['mAP']):+.2f} "
              f"macro-F1 {100 * (summary['test']['macro_F1'] - reference['macro_F1']):+.2f} points",
              flush=True)


if __name__ == "__main__":
    main()
