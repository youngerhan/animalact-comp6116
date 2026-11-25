# Image-Based Animal Action Recognition with ResNet-50

This practical introduces supervised animal action classification from a single RGB image. It covers four stages:

1. Data Loading and Preprocessing
2. Model Training
3. Model Evaluation
4. Prediction Visualization and Error Analysis

The notebook provides an image-based baseline for the video action-recognition project. It uses three actions: **Keeping still**, **Walking**, and **Eating**.

## Random Annotation Inspection

Open [annotation_spot_check.ipynb](annotation_spot_check.ipynb) to inspect video–annotation pairs without training a model. Run all cells once, then rerun its final code cell for a new random draw. Each pair includes the complete video, an action-interval table, and sampled frames labeled with all actions active at each timestamp.

The notebook checks file pairing, action vocabulary, video metadata, and interval boundaries. Visual review is still required to assess whether the labels match the behavior. Adjust `N_SAMPLES` and `FRAMES_PER_VIDEO`, or set `VIDEO_ID` to inspect a specific video. Dataset files are read without modification.

## Environment and Execution

Use Python with the packages listed in `requirements.txt`. The pinned environment was tested with Python 3.14. Activate a suitable existing environment, then run these commands from the `demo` directory:

```bash
python -m pip install -r requirements.txt
python -m ipykernel install --user --name animalact --display-name "Python (AnimalAct)"
python -m jupyterlab
```

Select **Python (AnimalAct)** and run notebook cells in order. Start with `fixed_split_usage.ipynb` to explore the published partitions, then open `animal_action_baseline.ipynb` for the image-classification exercise. `annotation_spot_check.ipynb` provides a video-and-label viewer.

On macOS, `start_jupyterlab.command` uses Python from the active environment. Run it from an activated terminal environment, or set `ANIMALACT_PYTHON` to the desired Python executable. It does not install packages or create environments. The terminal commands above also work on other platforms.

The first model run downloads approximately 98 MB of pretrained weights. Subsequent runs reuse the local cache. Feature extraction uses CUDA or Apple MPS when available, and otherwise uses the CPU. Runtime depends on the hardware.

## Data Preparation and Labels

Each entry in `manifest.json` links a video to its annotation through `video_id`, `video_path`, and `annotation_path`. For each video and selected action, the notebook extracts the midpoint frame of the longest span with one distinct active action and a duration of at least 0.5 seconds. The frame is saved as a PNG in `images/`.

The model uses the following local output indices:

| Output index | Action |
| ---: | --- |
| 0 | Keeping still |
| 1 | Walking |
| 2 | Eating |

These indices follow the notebook's `CLASS_NAMES` list. They differ from the dataset-wide IDs in [action_classes.json](../action_classes.json); see [ACTION_CLASSES.md](../ACTION_CLASSES.md) for the complete vocabulary.

A video may provide images for multiple actions. Its images remain in the same data split. Video IDs, timestamps, and annotations establish the image–label association; only image pixels enter the model. Source videos and annotations are read without modification.

## Model Input and Output

Each input is one RGB image, resized, center-cropped, and normalized to a tensor of shape `[3, 224, 224]`. A batch has shape `[N, 3, 224, 224]`.

The model uses ImageNet-1K V2 pretrained ResNet-50 with a frozen backbone. Global average pooling produces 2,048 features per image. Multiclass logistic regression fits a three-class linear head, and its coefficients are copied into the PyTorch model's final linear layer. The complete model outputs logits of shape `[N, 3]`.

`predict_image(image_path, model, preprocess, DEVICE, CLASS_NAMES)` accepts an image path and returns an action name and its softmax probability, for example:

```json
{"action": "Walking", "confidence": 0.72}
```

This is an illustrative output. The confidence is a model probability, not a guarantee of correctness. The classifier always selects one of the three supported actions; it does not detect unknown actions.

## Fixed Splits and Evaluation

The current dataset revision is `animalact-20261114-r2`, using split policy `animalact-70-15-15-v1`: 2,075 train, 446 valid and 447 test videos. All notebooks resolve `video_path` and `annotation_path` from the dataset root, including the split subdirectories. No local random split is created.

Start with [fixed_split_usage.ipynb](fixed_split_usage.ipynb) to load all 119 classes, build video-level multi-label targets and read the fixed **31-class** main-evaluation list. It defines an mAP function for real test predictions; it defines evaluation utilities without training a model. See [MAIN_EVALUATION.md](../MAIN_EVALUATION.md) for the full protocol.

The ResNet-50 notebook remains a three-class introductory image task. Its classes are part of the main list, but its Accuracy/Macro-F1 do not represent the full main evaluation. Every eligible image inherits its video's fixed split. An exact selected-image hash collision across splits stops execution instead of silently excluding or moving videos. This check does not detect near-duplicates.

Hyperparameters are selected on valid. The final head is fitted on train only, then evaluated on test. The report stores the split version, manifest hash, global class IDs, image mappings and actual counts. Image counts differ from video counts because only sufficiently long, unambiguous single-action spans are eligible. Metrics are per image, and images within a video can be correlated.

Runs write to `outputs_resnet50_fixed_v1/`. Each result records the manifest hash and class mapping so it can be matched to the dataset used. These generated files are created when you run the notebook.

The inspection notebook supports `SPLIT = None`, `"train"`, `"valid"`, or `"test"` for checking annotations in the released partitions.

## Generated Files

| Path | Contents |
| --- | --- |
| `images/` | Extracted RGB images in PNG format |
| `outputs_resnet50_fixed_v1/results.json` | Image–label mapping, splits, settings, predictions, metrics, and dependency versions |
| `outputs_resnet50_fixed_v1/resnet50_action.pt` | Fitted model state, class order, and preprocessing metadata |
| `outputs_resnet50_fixed_v1/confusion_matrix.png` | Test-set confusion matrix |
| `outputs_resnet50_fixed_v1/prediction_examples.png` | Correct and incorrect predictions |

Running the notebook again regenerates these outputs. The notebook includes instructions for restoring the saved model and applying it to another image.

## Project Extensions

For video action recognition, extend the baseline to use temporal information and report Accuracy, Macro-F1, and a confusion matrix under a clearly defined evaluation protocol.

For optional temporal video grounding, use an untrimmed video and a natural-language query to predict a matching time interval. Evaluate temporal IoU, mIoU, and R@1 at IoU thresholds 0.5 and 0.7, and specify how multiple valid reference intervals are handled.
