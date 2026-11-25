"""Predict AnimalAct actions in a trimmed video or a selected frame interval."""

import argparse
import io
import json
from pathlib import Path

import decord
import torch
from mmengine.config import Config
from mmaction.apis import init_recognizer

from clips import read_json, AnimalActClips
from runtime import DEFAULT_MODEL_CONFIG, PROJECT, video_logits
from train_strong import ClipDataset


def prediction_dataset(root, clip, classes, settings):
    """Use exactly the validation preprocessing of the checkpoint's training recipe."""
    if "eval_zoom" not in settings:
        return AnimalActClips(root, [clip], len(classes), settings["frames"],
                              settings["stride"], settings["image_size"],
                              sampling=settings.get("sampling", "fixed"))
    return ClipDataset(root, [clip], len(classes), settings["frames"],
                       settings["image_size"], False, eval_zoom=settings["eval_zoom"],
                       crop=settings.get("crop", "random"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--run", type=Path, default=PROJECT / "runs" / "pilot32-whole224")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, help="Exclusive end frame. Defaults to the video end.")
    options = parser.parse_args()

    video = options.video.resolve()
    reader = decord.VideoReader(io.BytesIO(video.read_bytes()), num_threads=1)
    end = len(reader) if options.end_frame is None else options.end_frame
    if not 0 <= options.start_frame < end <= len(reader):
        parser.error(f"Expected 0 <= start < end <= {len(reader)} for {video.name}.")
    del reader

    settings = read_json(options.run / "settings.json")
    best = torch.load(options.run / "best.pth", map_location="cpu", weights_only=True)
    classes = best["classes"]

    config = Config.fromfile(str(options.config or settings.get("config", DEFAULT_MODEL_CONFIG)))
    config.model.cls_head.num_classes = len(classes)
    model = init_recognizer(config, checkpoint=None, device="cuda:0")
    model.load_state_dict(best["state_dict"])
    clip = {
        "video_path": str(video),
        "start_frame": options.start_frame,
        "end_frame": end,
        "labels": [],
    }
    dataset = prediction_dataset(video.parent, clip, classes, settings)
    frames, _ = dataset[0]
    preprocessor = config.model.data_preprocessor
    mean = torch.tensor(preprocessor.mean, device="cuda").view(1, 3, 1, 1, 1)
    std = torch.tensor(preprocessor.std, device="cuda").view(1, 3, 1, 1, 1)
    with torch.inference_mode():
        scores = video_logits(model, frames.unsqueeze(0), mean, std)[0].sigmoid().cpu().tolist()
    ranked = sorted(({**label, "score": float(score)} for label, score in zip(classes, scores)),
                    key=lambda item: item["score"], reverse=True)
    output = {
        "video": str(video),
        "preprocessing": "training validation recipe (no evaluation-time ensemble)",
        "start_frame": options.start_frame,
        "end_frame_exclusive": end,
        "validation_threshold": best["threshold"],
        "predicted_actions": [label for label in ranked if label["score"] >= best["threshold"]],
        "top_scores": ranked[:5],
    }
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

