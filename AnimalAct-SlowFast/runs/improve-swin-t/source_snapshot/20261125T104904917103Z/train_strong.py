"""Fine-tune SlowFast R50 on AnimalAct with a stronger recipe.

Training options:
  * uniform temporal sampling over the annotated interval,
  * random-resized-crop, flip and colour jitter during training,
  * optional sqrt-inverse-frequency class-balanced sampling,
  * option to fine-tune the whole backbone,
  * warmup + cosine schedule, gradient clipping,
  * per-class thresholds reported next to the shared threshold (secondary metric only).
"""

import argparse
import io
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from clips import DEFAULT_DATASET, read_json, write_json
from metrics import evaluate, select_threshold, per_class_thresholds, summarise
from runtime import (PROJECT, DEFAULT_MODEL_CONFIG, DEFAULT_WEIGHTS, predict,
                   save_checkpoint, run_report, video_logits)
from provenance import record_run
from video_ops import context_interval, motion_box
from losses import AsymmetricLoss

SHORT_SECONDS = 1.0


# ---------------------------------------------------------------- data


class ClipDataset(Dataset):
    """Uniformly sampled clip with optional train-time augmentation."""

    def __init__(self, root, rows, class_count, frames, image_size, train,
                 scale=(0.8, 1.0), temporal=(0.75, 1.0), flip=0.5, colour=0.2,
                 eval_zoom=1.15, window=None, crop="random", crop_x=0.5, min_context_seconds=0.0,
                 spatial_view="whole", roi_probability=.7):
        self.root, self.rows, self.class_count = Path(root), rows, class_count
        self.frames, self.image_size, self.train = frames, image_size, train
        self.scale, self.temporal, self.flip, self.colour = scale, temporal, flip, colour
        self.eval_zoom = eval_zoom
        # (offset, span) as fractions of the clip, used only for evaluation-time windows.
        self.window = window
        # "random" crops a square out of a zoomed frame; "whole" squashes the full frame.
        self.crop, self.crop_x = crop, crop_x
        self.min_context_seconds = min_context_seconds
        self.spatial_view, self.roi_probability = spatial_view, roi_probability
        self.current_video, self.reader = None, None

    def __len__(self):
        return len(self.rows)

    def _frame_indices(self, start, end):
        length = max(end - start, 1)
        if not self.train and self.window is not None:
            offset, span = self.window
            first = start + int(round(length * offset))
            return self._uniform(first, max(1, int(round(length * span))))
        if self.train and length > self.frames:
            span = max(self.frames, int(round(length * random.uniform(*self.temporal))))
            span = min(span, length)
            start, length = start + random.randint(0, length - span), span
        return self._uniform(start, length)

    def _uniform(self, start, length):
        if length <= 1:
            return [start] * self.frames
        indices = np.linspace(start, start + length - 1, self.frames)
        return np.clip(np.round(indices).astype(int), start, start + length - 1).tolist()

    def _train_crop(self, height, width):
        side = max(1, int(round(min(height, width) * random.uniform(*self.scale))))
        top = random.randint(0, height - side)
        left = random.randint(0, width - side)
        return (width, height), (left, top, left + side, top + side)

    def _eval_crop(self, height, width):
        k = (self.image_size * self.eval_zoom) / min(height, width)
        resized_h, resized_w = int(round(height * k)), int(round(width * k))
        left = int(round((resized_w - self.image_size) * self.crop_x))
        top = (resized_h - self.image_size) // 2
        return (resized_w, resized_h), (left, top, left + self.image_size, top + self.image_size)

    def __getitem__(self, index):
        import cv2
        import decord

        row = self.rows[index]
        path = self.root / row["video_path"]
        if path != self.current_video:
            # decord cannot open non-ASCII Windows paths directly, so hand it a buffer.
            self.reader = decord.VideoReader(io.BytesIO(path.read_bytes()), num_threads=1)
            self.current_video = path
        if row["end_frame"] > len(self.reader):
            raise ValueError(f"Clip extends beyond decoded video: {path}")
        start, end = row["start_frame"], row["end_frame"]
        if self.min_context_seconds and self.train:
            start, end = context_interval(start, end, self.reader.get_avg_fps(),
                                          len(self.reader), self.min_context_seconds)
        frames = self.reader.get_batch(self._frame_indices(start, end)).asnumpy()
        if self.spatial_view == "motion" and (not self.train or random.random() < self.roi_probability):
            left, top, right, bottom = motion_box(frames)
            frames = frames[:, top:bottom, left:right]
        height, width = frames.shape[1:3]
        if self.crop == "whole":
            crops = [cv2.resize(frame, (self.image_size, self.image_size),
                                interpolation=cv2.INTER_AREA) for frame in frames]
        else:
            size, box = (self._train_crop(height, width) if self.train else self._eval_crop(height, width))
            crops = [cv2.resize(frame, size, interpolation=cv2.INTER_LINEAR)
                     [box[1]:box[3], box[0]:box[2]] for frame in frames]
        if self.train:
            output = self.image_size
            crops = [cv2.resize(frame, (output, output), interpolation=cv2.INTER_AREA)
                     if frame.shape[0] != output else frame for frame in crops]
            clip = np.stack(crops)
            if random.random() < self.flip:
                clip = clip[:, :, ::-1]
            if random.random() < self.colour:
                clip = clip.astype(np.float32) * random.uniform(1 - self.colour, 1 + self.colour) \
                    + random.uniform(-self.colour * 255 * 0.4, self.colour * 255 * 0.4)
                clip = np.clip(clip, 0, 255)
            video = torch.from_numpy(np.ascontiguousarray(clip)).to(torch.uint8)
        else:
            video = torch.from_numpy(np.stack(crops))
        video = video.permute(3, 0, 1, 2).contiguous()
        target = torch.zeros(self.class_count, dtype=torch.float32)
        target[row["labels"]] = 1
        return video, target


# ---------------------------------------------------------------- model


def build_model(options, class_count, weights_path):
    from mmaction.apis import init_recognizer

    model = init_recognizer(str(options.config), str(weights_path), device="cuda:0")
    head = model.cls_head.fc_cls
    model.cls_head.fc_cls = nn.Linear(head.in_features, class_count, device="cuda")
    nn.init.normal_(model.cls_head.fc_cls.weight, std=0.01)
    nn.init.zeros_(model.cls_head.fc_cls.bias)
    model.cls_head.num_classes = class_count

    model.backbone.requires_grad_(False)
    if options.trainable == "all":
        model.backbone.requires_grad_(True)
    else:
        for path in (model.backbone.slow_path, model.backbone.fast_path):
            if options.trainable == "l34":
                path.layer3.requires_grad_(True)
            path.layer4.requires_grad_(True)
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        [{"params": model.cls_head.parameters(), "lr": options.head_lr},
         {"params": backbone_params, "lr": options.backbone_lr}],
        weight_decay=options.weight_decay)
    return model, optimizer, backbone_params


def clip_weights(rows, class_count):
    counts = np.zeros(class_count)
    for row in rows:
        counts[row["labels"]] += 1
    weights = np.clip(np.sqrt((len(rows) - counts) / counts), 1, 6)
    return counts, weights


def sample_weights(rows, counts):
    inverse = 1.0 / np.sqrt(counts)
    return np.array([float(np.mean([inverse[label] for label in row["labels"]])) for row in rows])


# ---------------------------------------------------------------- metrics


# ---------------------------------------------------------------- loop


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--manifest", type=Path, default=PROJECT / "data" / "main.json")
    parser.add_argument("--config", type=Path, default=DEFAULT_MODEL_CONFIG)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--initial", type=Path, default=None,
                        help="Warm start from a saved best.pth instead of the Kinetics weights.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--keep-initial", action="store_true",
                        help="Include the warm-start checkpoint as epoch zero in validation selection.")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--schedule-epochs", type=int, default=0,
                        help="Cosine schedule horizon; defaults to --epochs. Lets short ablations "
                             "share the schedule of the full run.")
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--trainable", choices=("l4", "l34", "all"), default="all")
    parser.add_argument("--balance", choices=("none", "sqrt"), default="sqrt")
    parser.add_argument("--head-lr", type=float, default=5e-4)
    parser.add_argument("--backbone-lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--clip-grad", type=float, default=5.0)
    parser.add_argument("--crop", choices=("random", "whole"), default="random",
                        help="random: square crops of a zoomed frame; whole: squash the full frame.")
    parser.add_argument("--aug-scale", type=float, nargs=2, default=(0.8, 1.0))
    parser.add_argument("--aug-temporal", type=float, nargs=2, default=(0.75, 1.0))
    parser.add_argument("--colour", type=float, default=0.2)
    parser.add_argument("--eval-zoom", type=float, default=1.15)
    parser.add_argument("--deterministic", action="store_true",
                        help="Prefer deterministic CUDA operations; warn for unsupported operators.")
    parser.add_argument("--min-context-seconds", type=float, default=0.0,
                        help="Training-only context augmentation; evaluation stays inside the original clip.")
    parser.add_argument("--spatial-view", choices=("whole", "motion"), default="whole")
    parser.add_argument("--roi-probability", type=float, default=.7)
    parser.add_argument("--loss", choices=("bce", "asymmetric"), default="bce")
    parser.add_argument("--seed", type=int, default=42)
    options = parser.parse_args()
    if options.frames % 8 or min(options.frames, options.image_size, options.batch_size,
                                 options.epochs, options.workers + 1) <= 0:
        parser.error("Frames must be a multiple of 8, sizes positive and workers nonnegative.")
    if not 0 < options.aug_scale[0] <= options.aug_scale[1] <= 1.0:
        parser.error("--aug-scale must satisfy 0 < low <= high <= 1.")
    if not 0 < options.aug_temporal[0] <= options.aug_temporal[1] <= 1.0:
        parser.error("--aug-temporal must satisfy 0 < low <= high <= 1.")

    if not torch.cuda.is_available():
        parser.error("A CUDA GPU and the documented PyTorch environment are required.")
    if options.deterministic:
        import os
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
    if options.min_context_seconds < 0 or not 0 <= options.roi_probability <= 1:
        parser.error("Context must be nonnegative; ROI probability must be in [0,1].")
    torch.set_num_threads(2)
    torch.manual_seed(options.seed)
    torch.cuda.manual_seed_all(options.seed)
    np.random.seed(options.seed)
    random.seed(options.seed)
    torch.backends.cudnn.benchmark = False

    manifest = read_json(options.manifest)
    classes = manifest["classes"]
    train_rows, valid_rows = manifest["splits"]["train"], manifest["splits"]["valid"]
    settings = {key: str(value.resolve()) if isinstance(value, Path) else value
                for key, value in vars(options).items() if key not in {"resume", "test", "tta"}}
    if not options.deterministic:
        settings.pop("deterministic")
    if not options.keep_initial:
        settings.pop("keep_initial")
    if (options.min_context_seconds == 0 and options.spatial_view == "whole"
            and options.roi_probability == .7 and options.loss == "bce"):
        for key in ("min_context_seconds", "spatial_view", "roi_probability", "loss"):
            settings.pop(key)
    settings["aug_scale"] = list(options.aug_scale)
    settings["aug_temporal"] = list(options.aug_temporal)
    settings.update(classes=classes, dataset_revision=manifest["dataset_revision"],
                    split_version=manifest["split_version"])
    if options.resume:
        if read_json(options.out / "settings.json") != settings:
            raise ValueError("Resume settings differ.")
    else:
        options.out.mkdir(parents=True, exist_ok=False)
        write_json(options.out / "settings.json", settings)

    record_run(options.out, options)
    counts, positive_weights = clip_weights(train_rows, len(classes))
    write_json(options.out / "positive_weights.json",
               [{**label, "positive_weight": float(weight), "train_positives": int(count)}
                for label, weight, count in zip(classes, positive_weights, counts)])
    criterion = (AsymmetricLoss() if options.loss == "asymmetric" else nn.BCEWithLogitsLoss(
        pos_weight=None if options.balance == "sqrt" else
        torch.tensor(positive_weights, dtype=torch.float32, device="cuda")))

    total_epochs = options.schedule_epochs or options.epochs
    if total_epochs < options.epochs:
        parser.error("--schedule-epochs must be at least --epochs.")
    model, optimizer, backbone_params = build_model(options, len(classes), options.weights)
    if options.initial is not None and not options.resume:
        initial = torch.load(options.initial, map_location="cpu", weights_only=True)
        if initial["classes"] != classes:
            raise ValueError("Initial checkpoint uses a different class mapping.")
        model.load_state_dict(initial["state_dict"])
        print(f"Warm start from {options.initial} (epoch {initial['epoch']})", flush=True)
    per_epoch = steps_per_epoch(options, len(train_rows))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: warmup_cosine(step, options.warmup * per_epoch, total_epochs * per_epoch))
    scaler = torch.amp.GradScaler("cuda")
    mean = torch.tensor([123.675, 116.28, 103.53], device="cuda").view(1, 3, 1, 1, 1)
    std = torch.tensor([58.395, 57.12, 57.375], device="cuda").view(1, 3, 1, 1, 1)

    best_map, last_improved, first_epoch, history, step = -1.0, 0, 1, [], 0
    if options.resume:
        state = torch.load(options.out / "last.pth", map_location="cpu", weights_only=True)
        model.load_state_dict(state["state_dict"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        history, best_map = state["history"], state["best_map"]
        last_improved, first_epoch, step = state["last_improved"], state["epoch"] + 1, state["step"]

    if options.keep_initial and options.initial is not None and not options.resume:
        initial_loader = DataLoader(
            ClipDataset(options.dataset, valid_rows, len(classes), options.frames,
                        options.image_size, False, eval_zoom=options.eval_zoom, crop=options.crop,
                        min_context_seconds=options.min_context_seconds, spatial_view=options.spatial_view),
            batch_size=options.batch_size, num_workers=options.workers, pin_memory=True)
        initial_pairs = predict(model, initial_loader, mean, std)
        threshold = select_threshold(*initial_pairs)
        thresholds = per_class_thresholds(*initial_pairs)
        initial_report = run_report(options.out, "initial_valid", initial_pairs, classes,
                                    threshold, thresholds, valid_rows)
        best_map = initial_report["mAP"]
        save_checkpoint(options.out / "best.pth", dict(
            state_dict=model.state_dict(), classes=classes, epoch=0,
            threshold=threshold, thresholds=thresholds))
        history.append(dict(epoch=0, train_loss=None, valid_mAP=initial_report["mAP"],
                            valid_macro_F1=initial_report["macro_F1"], valid_threshold=threshold))
        print({"initial_validation": history[-1]}, flush=True)

    started = time.monotonic()
    for epoch in range(first_epoch, options.epochs + 1):
        random.seed(options.seed + epoch)
        np.random.seed(options.seed + epoch)
        torch.manual_seed(options.seed + epoch)
        train_dataset = ClipDataset(options.dataset, train_rows, len(classes), options.frames,
                                    options.image_size, True, options.aug_scale,
                                    options.aug_temporal, 0.5, options.colour, options.eval_zoom,
                                    crop=options.crop, min_context_seconds=options.min_context_seconds,
                                    spatial_view=options.spatial_view, roi_probability=options.roi_probability)
        sampler = None
        if options.balance == "sqrt":
            generator = torch.Generator().manual_seed(options.seed + epoch)
            sampler = WeightedRandomSampler(sample_weights(train_rows, counts), len(train_rows),
                                            replacement=True, generator=generator)
        train_loader = DataLoader(train_dataset, batch_size=options.batch_size, sampler=sampler,
                                  shuffle=sampler is None, num_workers=options.workers,
                                  pin_memory=True, drop_last=False)
        model.eval()
        model.cls_head.train()
        total_loss, examples, epoch_start = 0.0, 0, time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        for index, (videos, target) in enumerate(train_loader, start=1):
            target = target.cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = criterion(video_logits(model, videos, mean, std), target)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss in epoch {epoch}, batch {index}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(backbone_params + list(model.cls_head.parameters()),
                                                  options.clip_grad, error_if_nonfinite=False)
            if not torch.isfinite(norm):
                optimizer.zero_grad(set_to_none=True)
                scaler.update(new_scale=scaler.get_scale() / 2)
                continue
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            step += 1
            examples += len(target)
            total_loss += float(loss.detach()) * len(target)
            if index == 1 or index % 100 == 0 or index == len(train_loader):
                print(f"epoch {epoch} batch {index}/{len(train_loader)} "
                      f"loss={total_loss / examples:.4f} grad={float(norm):.2f} "
                      f"lr={optimizer.param_groups[0]['lr']:.2e} "
                      f"speed={examples / (time.monotonic() - epoch_start):.1f} clips/s", flush=True)

        valid_loader = DataLoader(ClipDataset(options.dataset, valid_rows, len(classes), options.frames,
                                              options.image_size, False, eval_zoom=options.eval_zoom,
                                              crop=options.crop, min_context_seconds=options.min_context_seconds,
                                    spatial_view=options.spatial_view, roi_probability=options.roi_probability),
                                  batch_size=options.batch_size, num_workers=options.workers,
                                  pin_memory=True)
        validation = predict(model, valid_loader, mean, std)
        threshold = select_threshold(*validation)
        thresholds = per_class_thresholds(*validation)
        report = evaluate(*validation, classes, threshold)
        summarise(report, *validation, thresholds)
        if not examples:
            raise FloatingPointError("No optimizer steps completed in this epoch.")
        short = np.asarray([row["duration_seconds"] <= SHORT_SECONDS for row in valid_rows])
        if short.any():
            report["short_clips"] = evaluate(validation[0][short], validation[1][short], classes, threshold)
        record = {"epoch": epoch, "train_loss": total_loss / examples, "valid_mAP": report["mAP"],
                  "valid_macro_F1": report["macro_F1"],
                  "valid_macro_F1_per_class_threshold": report["macro_F1_per_class_threshold"],
                  "valid_mAP_positives_ge_20": report["mAP_positives_ge_20"],
                  "valid_threshold": threshold,
                  "short_mAP": report.get("short_clips", {}).get("mAP"),
                  "gpu_peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 1024 ** 2),
                  "elapsed_minutes": round((time.monotonic() - started) / 60, 2)}
        history.append(record)
        print(record, flush=True)
        if report["mAP"] > best_map:
            best_map, last_improved = report["mAP"], epoch
            save_checkpoint(options.out / "best.pth", dict(
                state_dict=model.state_dict(), classes=classes, epoch=epoch,
                threshold=threshold, thresholds=thresholds))
        save_checkpoint(options.out / "last.pth", dict(
            state_dict=model.state_dict(), optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(), scaler=scaler.state_dict(), history=history,
            best_map=best_map, last_improved=last_improved, epoch=epoch, step=step))
        write_json(options.out / "history.json", history)
        if epoch - last_improved >= options.patience:
            print(f"Early stop: no validation mAP improvement for {options.patience} epochs.", flush=True)
            break

    # Keep validation artifacts paired with the selected checkpoint.
    best = torch.load(options.out / "best.pth", map_location="cpu", weights_only=True)
    model.load_state_dict(best["state_dict"])
    selected_loader = DataLoader(
        ClipDataset(options.dataset, valid_rows, len(classes), options.frames,
                    options.image_size, False, eval_zoom=options.eval_zoom, crop=options.crop, min_context_seconds=options.min_context_seconds,
                                    spatial_view=options.spatial_view, roi_probability=options.roi_probability),
        batch_size=options.batch_size, num_workers=options.workers, pin_memory=True)
    selected_validation = predict(model, selected_loader, mean, std)
    run_report(options.out, "valid", selected_validation, classes, best["threshold"],
               best["thresholds"], valid_rows)

    if options.test:
        evaluate_test(options, manifest, classes, mean, std)


def steps_per_epoch(options, rows):
    return math.ceil(rows / options.batch_size)


def warmup_cosine(step, warmup_steps, total_steps, eta_min_ratio=0.02):
    if step < warmup_steps:
        return (step + 1) / max(warmup_steps, 1)
    progress = min((step - warmup_steps) / max(total_steps - warmup_steps, 1), 1.0)
    return eta_min_ratio + (1 - eta_min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))


def evaluate_test(options, manifest, classes, mean, std):
    best = torch.load(options.out / "best.pth", map_location="cpu", weights_only=True)
    model, _, _ = build_model(options, len(classes), options.weights)
    model.load_state_dict(best["state_dict"])
    rows = manifest["splits"]["test"]
    loader = DataLoader(ClipDataset(options.dataset, rows, len(classes), options.frames,
                                    options.image_size, False, eval_zoom=options.eval_zoom,
                                    crop=options.crop, min_context_seconds=options.min_context_seconds,
                                    spatial_view=options.spatial_view, roi_probability=options.roi_probability),
                        batch_size=options.batch_size, num_workers=options.workers, pin_memory=True)
    pairs = predict(model, loader, mean, std)
    report = run_report(options.out, "test", pairs, classes, best["threshold"],
                        best["thresholds"], rows)
    print(f"test mAP={report['mAP']:.4f} macro-F1={report['macro_F1']:.4f} "
          f"macro-F1(per-class thr)={report['macro_F1_per_class_threshold']:.4f} "
          f"mAP(pos>=20)={report['mAP_positives_ge_20']} "
          f"epoch={best['epoch']}", flush=True)


@torch.inference_mode()
def predict_tta(model, loader, mean, std):
    """Average the probability of the clip and its horizontal mirror."""
    model.eval()
    scores, targets = [], []
    for video, target in loader:
        logits = video_logits(model, video, mean, std).sigmoid()
        flipped = video_logits(model, video.flip(-1), mean, std).sigmoid()
        scores.append(((logits + flipped) / 2).cpu().numpy())
        targets.append(target.numpy().astype(bool))
    return np.concatenate(targets), np.concatenate(scores)


if __name__ == "__main__":
    main()
