"""Build a clip manifest over the full 119-class vocabulary.

The course protocol allows training with the full vocabulary and reporting mAP over the fixed
main classes. The extra 88 classes are not scored, but they turn clips that the main-only
manifest throws away into labelled training examples and supply harder negatives.
"""

import argparse
from collections import Counter
from pathlib import Path

from clips import DEFAULT_DATASET, LABEL_OVERLAP, SPLITS, clip_rows, read_json, write_json


def prepare(root, output):
    vocabulary = read_json(root / "action_classes.json")
    classes = [{"global_id": item["class_id"], "name": item["name"]}
               for item in sorted(vocabulary["classes"], key=lambda item: item["class_id"])]
    class_indices = {item["name"]: index for index, item in enumerate(classes)}
    main = read_json(root / "main_evaluation_classes.json")
    splits, counts = {}, {}
    for split in SPLITS:
        official = read_json(root / "splits" / f"{split}.json")
        if official["split_version"] != main["split_version"]:
            raise ValueError(f"Inconsistent split version in {split}")
        rows, corrections = [], Counter()
        for sample in official["samples"]:
            annotation = read_json(root / sample["annotation_path"])
            if sample["split"] != split or annotation["video_id"] != sample["video_id"]:
                raise ValueError(f"Split or annotation mismatch for {sample['video_id']}")
            rows.extend(clip_rows(sample, annotation, class_indices, False, corrections))
        frequencies = Counter(label for row in rows for label in row["labels"])
        counts[split] = {
            "frame_boundary_corrections": dict(corrections),
            "clips": len(rows),
            "videos": len({row["video_id"] for row in rows}),
            "multi_label_clips": sum(len(row["labels"]) > 1 for row in rows),
            "support": {entry["name"]: frequencies[index] for index, entry in enumerate(classes)},
        }
        splits[split] = rows
    write_json(output, {
        "split_version": main["split_version"],
        "dataset_revision": main["dataset_revision"],
        "mode": "main",
        "label_overlap_fraction_of_window": LABEL_OVERLAP,
        "main_class_ids": main["class_ids"],
        "classes": classes,
        "counts": counts,
        "splits": splits,
    })
    return counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "data" / "main_all.json")
    options = parser.parse_args()
    for partition, stats in prepare(options.dataset, options.output).items():
        unsupported = [name for name, count in stats["support"].items() if count == 0]
        print(f"{partition}  {stats['clips']} clips, {stats['multi_label_clips']} multi-label, "
              f"{len(unsupported)} classes without clips", flush=True)
