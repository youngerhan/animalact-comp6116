"""Average the clip scores of several runs and report the ensemble on validation and test.

Members must share the manifest, so clip order and targets match exactly.
"""

import argparse
from pathlib import Path

import numpy as np

from clips import read_json, write_json
from metrics import evaluate, select_threshold
from runtime import PROJECT
from metrics import per_class_thresholds, summarise


from predictions import combine


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, default=PROJECT / "runs" / "ensemble")
    parser.add_argument("--allow-legacy", action="store_true",
                        help="Allow historical files with incomplete identity metadata.")
    options = parser.parse_args()
    classes = read_json(PROJECT / "data" / "main.json")["classes"]

    valid = combine(options.runs, "valid", allow_legacy=options.allow_legacy)
    threshold = select_threshold(*valid)
    thresholds = per_class_thresholds(*valid)
    valid_report = summarise(evaluate(*valid, classes, threshold), *valid, thresholds)
    print(f"ensemble valid  mAP={valid_report['mAP']:.4f}  "
          f"macro-F1={valid_report['macro_F1']:.4f}  "
          f"macro-F1(per-class)={valid_report['macro_F1_per_class_threshold']:.4f}", flush=True)

    test = combine(options.runs, "test", allow_legacy=options.allow_legacy)
    test_report = summarise(evaluate(*test, classes, threshold), *test, thresholds)
    manifest = read_json(PROJECT / "data" / "main.json")
    short = np.asarray([row["duration_seconds"] <= 1.0 for row in manifest["splits"]["test"]])
    if short.any():
        test_report["short_clips"] = {"maximum_seconds": 1.0,
                                      **evaluate(test[0][short], test[1][short], classes, threshold)}
    print(f"ensemble test   mAP={test_report['mAP']:.4f}  "
          f"macro-F1={test_report['macro_F1']:.4f}  "
          f"macro-F1(per-class)={test_report['macro_F1_per_class_threshold']:.4f}", flush=True)

    options.out.mkdir(parents=True, exist_ok=False)
    for split, pairs in (("valid", valid), ("test", test)):
        extra = {}
        with np.load(options.runs[0] / f"{split}_predictions.npz") as source:
            for key in ("global_class_ids", "video_ids", "start_frames", "end_frames"):
                if key in source.files:
                    extra[key] = source[key]
        np.savez_compressed(options.out / f"{split}_predictions.npz", targets=pairs[0],
                            scores=pairs[1], thresholds=np.asarray(thresholds), **extra)
    write_json(options.out / "valid_metrics.json", valid_report)
    write_json(options.out / "test_metrics.json", test_report)
    write_json(options.out / "members.json", {
        "legacy_identity_unverified": options.allow_legacy,
        "members": [str(run) for run in options.runs], "threshold": threshold,
        "per_class_thresholds": thresholds,
        "valid": {"mAP": valid_report["mAP"], "macro_F1": valid_report["macro_F1"],
                  "macro_F1_per_class_threshold": valid_report["macro_F1_per_class_threshold"],
                  "mAP_positives_ge_20": valid_report["mAP_positives_ge_20"]},
        "test": {"mAP": test_report["mAP"], "macro_F1": test_report["macro_F1"],
                 "macro_F1_per_class_threshold": test_report["macro_F1_per_class_threshold"],
                 "mAP_positives_ge_10": test_report["mAP_positives_ge_10"],
                 "mAP_positives_ge_20": test_report["mAP_positives_ge_20"]}})
    print(f"Saved to {options.out}", flush=True)


if __name__ == "__main__":
    main()
