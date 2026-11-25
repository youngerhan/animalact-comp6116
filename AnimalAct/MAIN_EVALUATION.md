# Main Evaluation Classes

Split version: `animalact-70-15-15-v1`. The dataset retains all 119 classes; 31 classes form the fixed main-evaluation set.

A main class must have at least 20 detected groups in total, at least one training group, and at least three groups in each of validation and test. These thresholds provide minimum support, not a guarantee of statistical stability. Selection uses annotation and group counts, not model performance. Global class IDs remain unchanged.

## Evaluation Protocol

- For video-level multi-label classification, train with the full vocabulary if desired and report mean average precision (mAP) over the fixed main classes. Also report per-class AP and test positive counts. Align prediction columns with global class IDs. Do not discard other labels or force multi-label videos into a single class.
- Select hyperparameters and classification thresholds on validation data. If reporting Macro-F1, use the fixed main-class list, document the threshold rule, and assign F1 = 0 when there are no predicted positives.
- AP is not evaluable for an extension class with no test positives. Report this explicitly rather than treating its AP as zero in an all-class mean. Keep extension metrics separate from the main metric.
- The three-class image demo uses Keeping still, Walking, and Eating. Its Accuracy and Macro-F1 describe an introductory image task, not overall performance across 119 classes.
- Video classification, action-clip classification, and temporal localization use different evaluation units. Their scores are not directly comparable merely because they share splits. Temporal localization requires explicit IoU and matching rules, including the handling of multiple labels.
- If task-specific sampling or filtering leaves a main class unsupported, report that limitation rather than silently changing the main-class list.

Machine-readable definition: [main_evaluation_classes.json](main_evaluation_classes.json).

| Global ID | Action | Train groups | Valid groups | Test groups |
|---:|---|---:|---:|---:|
| 1 | Attacking | 27 | 5 | 6 |
| 2 | Attending | 213 | 52 | 51 |
| 7 | Being Eaten | 16 | 3 | 4 |
| 8 | Biting | 35 | 8 | 8 |
| 13 | Carrying In Mouth | 20 | 4 | 4 |
| 15 | Chirping | 157 | 41 | 39 |
| 16 | Climbing | 24 | 5 | 5 |
| 26 | Displaying Defensive Pose | 14 | 3 | 3 |
| 27 | Disturbing Another Animal | 26 | 6 | 6 |
| 32 | Doing A Face Dip | 21 | 4 | 3 |
| 37 | Drinking | 37 | 9 | 8 |
| 39 | Eating | 213 | 47 | 47 |
| 44 | Exploring | 40 | 9 | 8 |
| 46 | Fighting | 36 | 8 | 7 |
| 47 | Flapping | 40 | 9 | 8 |
| 50 | Flying | 94 | 22 | 21 |
| 53 | Grooming | 23 | 5 | 5 |
| 59 | Hugging | 14 | 4 | 3 |
| 61 | Jumping | 64 | 14 | 14 |
| 62 | Keeping still | 497 | 118 | 115 |
| 63 | Landing | 47 | 11 | 10 |
| 70 | Moving | 265 | 58 | 59 |
| 76 | Preening | 64 | 16 | 14 |
| 85 | Running | 65 | 13 | 14 |
| 87 | Sensing | 461 | 108 | 107 |
| 88 | Shaking | 20 | 5 | 5 |
| 89 | Shaking Head | 64 | 15 | 15 |
| 96 | Spitting Venom | 28 | 6 | 6 |
| 106 | Swimming | 40 | 10 | 9 |
| 110 | Turning Around | 21 | 6 | 5 |
| 114 | Walking | 286 | 67 | 66 |

`Washing` has 22 videos but 19 detected groups, so it is outside the main-evaluation set. Its annotations and global class ID remain available.
