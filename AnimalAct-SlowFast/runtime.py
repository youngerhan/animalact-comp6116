"""Shared model execution, checkpoint and report helpers; importing does not run jobs."""
import csv
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from clips import write_json
from metrics import evaluate, summarise

PROJECT = Path(__file__).resolve().parent
DEFAULT_MODEL_CONFIG = PROJECT.parent / "mmaction2-main" / "configs" / "recognition" / "slowfast" / (
    "slowfast_r50_8xb8-4x16x1-256e_kinetics400-rgb.py")
DEFAULT_WEIGHTS = PROJECT.parent / "slowfast_r50_8xb8-4x16x1-256e_kinetics400-rgb_20220901-701b0f6f.pth"
SHORT_SECONDS = 1.0


def video_logits(model, videos, mean, std):
    videos = videos.to(device="cuda", dtype=torch.float32, non_blocking=True)
    videos.sub_(mean).div_(std)
    with torch.autocast("cuda", dtype=torch.float16):
        return model.cls_head(model.backbone(videos)).float()


def train_epoch(model, loader, optimizer, scaler, mean, std, epoch, log_every):
    model.eval()  # Keep pretrained batch-norm statistics fixed for small batches.
    model.cls_head.train()
    criterion = nn.BCEWithLogitsLoss()
    start = time.monotonic()
    total_loss = 0.0
    examples = 0
    for step, (videos, targets) in enumerate(loader, start=1):
        targets = targets.cuda(non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(video_logits(model, videos, mean, std), targets)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at epoch {epoch}, batch {step}")
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        examples += len(videos)
        total_loss += float(loss.detach()) * len(videos)
        if step == 1 or step % log_every == 0 or step == len(loader):
            elapsed = time.monotonic() - start
            print(f"epoch {epoch}  batch {step}/{len(loader)}  loss={total_loss/examples:.4f}  "
                  f"{examples/elapsed:.1f} clips/s", flush=True)
    return total_loss / examples


@torch.inference_mode()
def predict(model, loader, mean, std):
    model.eval()
    scores, targets = [], []
    for video, target in loader:
        scores.append(video_logits(model, video, mean, std).sigmoid().cpu().numpy())
        targets.append(target.numpy().astype(bool))
    return np.concatenate(targets), np.concatenate(scores)


def save_predictions(out, name, pairs, report, rows):
    targets, scores = pairs
    np.savez_compressed(out / f"{name}_predictions.npz", scores=scores, targets=targets,
                        global_class_ids=[entry["global_id"] for entry in report["per_class"]],
                        video_ids=[row["video_id"] for row in rows],
                        start_frames=[row["start_frame"] for row in rows],
                        end_frames=[row["end_frame"] for row in rows])
    write_json(out / f"{name}_metrics.json", report)
    with (out / f"{name}_per_class.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(report["per_class"][0]))
        writer.writeheader()
        writer.writerows(report["per_class"])


def save_checkpoint(path, state):
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def save_split_report(out, name, pairs, classes, threshold, rows):
    report = evaluate(*pairs, classes, threshold)
    short = np.asarray([row["duration_seconds"] <= SHORT_SECONDS for row in rows])
    if short.any():
        subset = evaluate(pairs[0][short], pairs[1][short], classes, threshold)
        report["short_clips"] = {"maximum_seconds": SHORT_SECONDS, **subset}
    save_predictions(out, name, pairs, report, rows)
    return report


def run_report(out, name, pairs, classes, threshold, thresholds, rows):
    report = evaluate(*pairs, classes, threshold)
    short = np.asarray([row["duration_seconds"] <= SHORT_SECONDS for row in rows])
    if short.any():
        report["short_clips"] = {"maximum_seconds": SHORT_SECONDS,
                                 **evaluate(pairs[0][short], pairs[1][short], classes, threshold)}
    summarise(report, *pairs, thresholds)
    save_predictions(out, name, pairs, report, rows)
    return report


