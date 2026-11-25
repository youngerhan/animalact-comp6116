# AnimalAct Dataset

AnimalAct provides animal videos with action labels and temporal intervals. This student edition contains **2,968 videos**, **7,123 action intervals**, and **119 action classes**.

- Dataset revision: `animalact-20261114-r2`
- Vocabulary version: `1.0.0`
- Fixed split version: `animalact-70-15-15-v1`

## Getting Started

1. Extract the archive and keep the `AnimalAct` directory structure intact.
2. Read [demo/README.md](demo/README.md) for environment setup.
3. Open [fixed_split_usage.ipynb](demo/fixed_split_usage.ipynb) to load the fixed partitions and video-level multi-label targets.
4. Run [animal_action_baseline.ipynb](demo/animal_action_baseline.ipynb) for a three-class image-classification exercise, or [annotation_spot_check.ipynb](demo/annotation_spot_check.ipynb) to explore videos and their annotations.

All notebooks are supplied without saved outputs. They generate their own results when run.

## Directory Structure

```text
AnimalAct/
├── videos/{train,valid,test}/{video_id}.mp4
├── annotations/{train,valid,test}/{video_id}.json
├── manifest.json
├── action_classes.json
├── main_evaluation_classes.json
├── ACTION_CLASSES.md
├── MAIN_EVALUATION.md
├── splits/
│   ├── split_manifest.json
│   ├── assignments.csv
│   ├── train.json / valid.json / test.json
│   ├── train.txt / valid.txt / test.txt
│   ├── class_statistics.csv
│   ├── duplicate_audit.json
│   └── verification_report.json
├── demo/
│   ├── README.md
│   ├── requirements.txt
│   ├── fixed_split_usage.ipynb
│   ├── animal_action_baseline.ipynb
│   ├── annotation_spot_check.ipynb
│   └── start_jupyterlab.command
├── tools/validate_dataset.py
├── SHA256SUMS.txt
└── README.md
```

Each video is paired with one annotation file through `video_id`. For example, `videos/valid/AJDDHWBO.mp4` corresponds to `annotations/valid/AJDDHWBO.json`. JSON files use UTF-8 encoding. Paths stored in manifests are relative to the dataset root, including paths in `splits/*.json`.

## Fixed Partitions

| Split | Videos | Action intervals | Purpose |
|---|---:|---:|---|
| train | 2,075 | 4,977 | Fit models and learn training statistics |
| valid | 446 | 1,055 | Select hyperparameters, thresholds, and checkpoints |
| test | 447 | 1,091 | Evaluate the final selected model |

Use the published lists rather than making a new random split. Every derived frame, action clip, crop, or augmented sample inherits its source video's split. Fit on train, select settings on valid, and evaluate on test; do not merge validation data into training for the standard exercise.

The initial partition aimed for approximately 70%/15%/15% of videos. Class proportions are not guaranteed to match those ratios exactly. The published manifests define the current membership. All 119 classes occur in training. The fixed **31-class main-evaluation set** and its support counts are described in [MAIN_EVALUATION.md](MAIN_EVALUATION.md).

Videos sharing identical file hashes or non-uniform sampled frames are assigned to the same detected group. Grouping checks include five uniformly spaced frames per video and the image-demo sampling positions. This reduces detectable exact duplication; it does not establish complete independence of original recordings, scenes, or individual animals, or detect all near-duplicates. Original recording identifiers are unavailable.

## Loading Samples

Run the following code from the dataset root:

```python
import json
from pathlib import Path

root = Path(".")
train = json.loads((root / "splits/train.json").read_text(encoding="utf-8"))["samples"]
for sample in train:
    video_path = root / sample["video_path"]
    annotation = json.loads(
        (root / sample["annotation_path"]).read_text(encoding="utf-8")
    )
```

Use `valid`, not `val`, when selecting the validation split.

## Dataset Manifest

`manifest.json` is the dataset entry point. Its `count` is the total sample count, and `samples` contains one record per video.

| Field | Meaning |
|---|---|
| `video_id` | Unique ID shared by the video and annotation filenames |
| `video` / `annotation` | Video and annotation filenames |
| `video_path` / `annotation_path` | Paths relative to the dataset root |
| `split` | Published partition: `train`, `valid`, or `test` |
| `split_group_id` | Connected group detected from identical files or sampled frames; not an original recording ID |
| `parent_class` | Broad animal category |
| `duration` | Video duration in seconds |
| `fps` | Frames per second |
| `nb_frames` | Number of video frames |
| `source_labels` | Candidate `[animal, action]` pairs from source metadata |
| `n_labels` | Number of candidate source labels |
| `action_count` | Number of annotated action intervals across all animals |
| `video_sha256` / `annotation_sha256` | File checksums |

**Use the annotation's `annotations` array for actual labels and intervals.** `source_labels` and `n_labels` describe candidate metadata and are not substitutes for the annotations.

## Annotation Format

An annotation contains a video identifier, frame rate, animal metadata, and action intervals grouped by animal. For example:

```json
{
  "video_id": "AJDDHWBO",
  "video": "AJDDHWBO.mp4",
  "fps": 24,
  "list_animal": ["Golden Eagle"],
  "list_animal_action": [["Golden Eagle", "Walking"]],
  "source_labels": [["Golden Eagle", "Walking"]],
  "annotations": [
    {
      "animal": "Golden Eagle",
      "actions": [
        {
          "action": "Walking",
          "start": 0.941,
          "end": 7.0,
          "start_frame": 23,
          "end_frame": 168
        }
      ]
    }
  ]
}
```

| Field | Meaning |
|---|---|
| `video_id` / `video` | Paired video ID and filename |
| `fps` | Annotation frame rate |
| `list_animal` | Animal names |
| `list_animal_action` | Animal–action pairs |
| `source_labels` | Candidate source labels |
| `annotations[].animal` | Animal associated with the action intervals |
| `annotations[].actions[].action` | Case-sensitive action name |
| `start` / `end` | Start and end times in seconds from the video beginning |
| `start_frame` / `end_frame` | Optional frame boundaries corresponding to the times |

Multiple animals and overlapping action intervals are allowed. Treat the annotations as multi-label temporal data rather than assuming one mutually exclusive label per video. The example notebooks display active actions using `start <= time < end`.

Frame boundaries start at zero. `end_frame` may equal `nb_frames` to indicate the end of the video; it is not then a valid frame index. Limit decoded frame indices to `0 ... nb_frames - 1`. Timestamp rounding may differ from frame boundaries by up to one frame. Annotation entries are not spatial boxes or individual tracking IDs.

## Action Vocabulary and Evaluation

[action_classes.json](action_classes.json) maps action names to stable global IDs. [ACTION_CLASSES.md](ACTION_CLASSES.md) provides the readable table. Global IDs are zero-based and remain unchanged in this edition.

| Vocabulary field | Meaning |
|---|---|
| `num_classes` | Number of action classes |
| `classes[].class_id` / `name` | Global ID and exact action name |
| `interval_count` / `video_count` | Action records / distinct videos containing the action |
| `train_*`, `valid_*`, `test_*` | Counts in each fixed partition |
| `group_count` | Detected groups containing the action |
| `splits` / `split_coverage` | Partitions in which the class occurs |
| `main_evaluation` | Whether the class belongs to the fixed main-evaluation set |

A video can contain multiple classes, so per-class video counts do not sum to the dataset size. When training a subset of classes, save the mapping from local classifier indices to global class IDs.

The image demo uses Keeping still, Walking, and Eating with local indices 0, 1, and 2. It reports Accuracy, Macro-F1, and a confusion matrix for sampled images. Those results do not measure performance on the full video-level multi-label task. See [MAIN_EVALUATION.md](MAIN_EVALUATION.md) for the latter's fixed class set and mAP protocol.

## File Verification

From any working directory, run:

```bash
python /path/to/AnimalAct/tools/validate_dataset.py
```

The validator requires only the Python standard library. It checks media and annotation hashes, paths, partition membership, detected groups, interval bounds, class statistics, and main-evaluation lists. Structural checks do not certify the semantic correctness of every annotation.

`SHA256SUMS.txt` lists checksums for every other file in this package. On macOS, run `shasum -a 256 -c SHA256SUMS.txt` from the dataset root; on Linux, use `sha256sum -c SHA256SUMS.txt`.
