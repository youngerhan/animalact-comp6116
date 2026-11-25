"""Build and load clip manifests from the official AnimalAct split lists."""

import argparse
import io
import json
from collections import Counter, defaultdict
from pathlib import Path

SMOKE_NAMES = ("Keeping still", "Walking", "Eating")
SPLITS = ("train", "valid", "test")
DEFAULT_DATASET = Path(__file__).resolve().parent.parent / "AnimalAct"
LABEL_OVERLAP = 0.5


def read_json(path):
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    # Bundle-only portability: run paths are stored relative to the project root.
    if path.name == "settings.json" and isinstance(data, dict):
        base = Path(__file__).resolve().parent
        for key in ("dataset", "manifest", "config", "weights", "out", "initial"):
            value = data.get(key)
            if isinstance(value, str) and not Path(value).is_absolute():
                data[key] = str((base / value).resolve())
    return data


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")
    temp.replace(path)


def annotation_intervals(annotation, frame_count, fps, corrections):
    intervals = set()
    for animal in annotation["annotations"]:
        for action in animal["actions"]:
            start = action.get("start_frame")
            end = action.get("end_frame")
            if start is None:
                start = round(float(action["start"]) * fps)
            if end is None:
                end = round(float(action["end"]) * fps)
            start, end = int(start), int(end)
            if end == frame_count + 1:
                end = frame_count
                corrections["end_clamped_one_frame"] += 1
            if start == end and 0 <= start < frame_count and float(action["end"]) > float(action["start"]):
                end = start + 1
                corrections["subframe_interval_expanded"] += 1
            if not 0 <= start < end <= frame_count:
                raise ValueError(f"Invalid interval in {annotation['video_id']}: {start}:{end}/{frame_count}")
            intervals.add((start, end, action["action"]))
    return sorted(intervals)


def clip_rows(sample, annotation, class_indices, smoke, corrections):
    fps = float(sample["fps"])
    intervals = annotation_intervals(annotation, int(sample["nb_frames"]), fps, corrections)
    windows = sorted({(start, end) for start, end, name in intervals if name in class_indices})
    rows = []
    for start, end in windows:
        overlap_frames = Counter()
        last_end_by_action = defaultdict(lambda: start)
        length = end - start
        for left, right, name in intervals:
            overlap_start, overlap_end = max(start, left), min(end, right)
            if overlap_start < overlap_end:
                previous_end = last_end_by_action[name]
                overlap_frames[name] += max(0, overlap_end - max(overlap_start, previous_end))
                last_end_by_action[name] = max(previous_end, overlap_end)
        if smoke and (len(overlap_frames) != 1 or length / fps < 0.5):
            continue
        labels = sorted(class_indices[name] for name, overlap in overlap_frames.items()
                        if name in class_indices and overlap >= LABEL_OVERLAP * length)
        rows.append({
            "video_id": sample["video_id"],
            "video_path": sample["video_path"],
            "start_frame": start,
            "end_frame": end,
            "duration_seconds": length / fps,
            "labels": labels,
        })
    return rows


def prepare(root, output, smoke=False):
    main = read_json(root / "main_evaluation_classes.json")
    classes = [{"global_id": class_id, "name": name} for class_id, name in
               zip(main["class_ids"], main["class_names"]) if not smoke or name in SMOKE_NAMES]
    class_indices = {item["name"]: index for index, item in enumerate(classes)}
    splits, counts = {}, {}
    for split in SPLITS:
        official = read_json(root / "splits" / f"{split}.json")
        if official["split_version"] != main["split_version"]:
            raise ValueError(f"Inconsistent split version in {split}")
        rows = []
        corrections = Counter()
        for sample in official["samples"]:
            annotation = read_json(root / sample["annotation_path"])
            if sample["split"] != split or annotation["video_id"] != sample["video_id"]:
                raise ValueError(f"Split or annotation mismatch for {sample['video_id']}")
            rows.extend(clip_rows(sample, annotation, class_indices, smoke, corrections))
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
        "mode": "smoke" if smoke else "main",
        "label_overlap_fraction_of_window": LABEL_OVERLAP,
        "classes": classes,
        "counts": counts,
        "splits": splits,
    })
    return counts


def frame_indices(start, end, frame_count, stride):
    """Center a fixed-stride sample inside the clip, repeating the last frame as needed."""
    required = 1 + (frame_count - 1) * stride
    first = start + max(0, ((end - start) - required) // 2)
    return [min(first + stride * index, end - 1) for index in range(frame_count)]


class AnimalActClips:
    def __init__(self, root, rows, classes, frame_count, stride, image_size, sampling='fixed'):
        self.root = root
        self.rows = rows
        self.classes = classes
        self.frame_count = frame_count
        self.stride = stride
        self.image_size = image_size
        if sampling not in ('fixed', 'uniform'):
            raise ValueError(f'Unsupported sampling policy: {sampling}')
        self.sampling = sampling
        self.current_video = None
        self.reader = None

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        import cv2
        import decord
        import numpy as np
        import torch

        row = self.rows[index]
        path = self.root / row["video_path"]
        if path != self.current_video:
            self.reader = decord.VideoReader(io.BytesIO(path.read_bytes()), num_threads=1)
            self.current_video = path
        if row["end_frame"] > len(self.reader):
            raise ValueError(f"Clip extends beyond decoded video: {path}, {row['end_frame']} > {len(self.reader)}")
        if self.sampling == "uniform":
            indices = np.linspace(row["start_frame"], row["end_frame"] - 1, self.frame_count).round().astype(int).tolist()
        else:
            indices = frame_indices(row["start_frame"], row["end_frame"], self.frame_count, self.stride)
        frames = self.reader.get_batch(indices).asnumpy()
        height, width = frames.shape[1:3]
        scale = self.image_size / min(height, width)
        resized_width, resized_height = round(width * scale), round(height * scale)
        x = (resized_width - self.image_size) // 2
        y = (resized_height - self.image_size) // 2
        crops = [cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
                 [y:y + self.image_size, x:x + self.image_size] for frame in frames]
        video = torch.from_numpy(np.stack(crops)).permute(3, 0, 1, 2).contiguous()
        target = torch.zeros(self.classes, dtype=torch.float32)
        target[row["labels"]] = 1
        return video, target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    options = parser.parse_args()
    for partition, stats in prepare(options.dataset, options.output, options.smoke).items():
        unsupported = [name for name, count in stats["support"].items() if count == 0]
        print(f"{partition}  {stats['clips']} clips, {stats['multi_label_clips']} multi-label, "
              f"zero support: {unsupported}", flush=True)

