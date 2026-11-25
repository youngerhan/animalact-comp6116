"""Train the official SlowFast R50 on AnimalAct action clips."""

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from clips import AnimalActClips, DEFAULT_DATASET, read_json, write_json
from metrics import evaluate, select_threshold
from runtime import (PROJECT, DEFAULT_MODEL_CONFIG, DEFAULT_WEIGHTS, SHORT_SECONDS,
                     video_logits, predict, train_epoch, save_predictions, save_checkpoint, save_split_report)
from provenance import record_run

def args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--manifest", type=Path, default=PROJECT / "data" / "main.json")
    parser.add_argument("--config", type=Path, default=DEFAULT_MODEL_CONFIG)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--head-lr", type=float, default=3e-4)
    parser.add_argument("--backbone-lr", type=float, default=3e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--test", action="store_true",
                        help="Evaluate the validation-selected checkpoint once on the official test split.")
    parser.add_argument("--resume", action="store_true", help="Resume from last.pth in the same output folder.")
    options = parser.parse_args()
    if options.frames < 8 or options.frames % 8:
        parser.error("--frames must be a positive multiple of 8 for this SlowFast config.")
    if min(options.epochs, options.batch_size, options.stride, options.image_size, options.log_every) <= 0:
        parser.error("Epochs, batch size, stride, spatial size and log frequency must be positive.")
    if options.workers < 0 or options.smoke and options.test:
        parser.error("Workers must be nonnegative, and smoke runs use only train and validation.")
    return options


def make_model(options, class_count):
    from mmaction.apis import init_recognizer

    model = init_recognizer(str(options.config), str(options.weights), device="cuda:0")
    old_head = model.cls_head.fc_cls
    model.cls_head.fc_cls = nn.Linear(old_head.in_features, class_count, device="cuda")
    nn.init.normal_(model.cls_head.fc_cls.weight, std=0.01)
    nn.init.zeros_(model.cls_head.fc_cls.bias)
    model.cls_head.num_classes = class_count

    model.backbone.requires_grad_(False)
    trainable_stages = (model.backbone.slow_path.layer4, model.backbone.fast_path.layer4)
    for stage in trainable_stages:
        stage.requires_grad_(True)
    optimizer = torch.optim.AdamW(
        [{"params": model.cls_head.parameters(), "lr": options.head_lr},
         {"params": [p for stage in trainable_stages for p in stage.parameters()], "lr": options.backbone_lr}],
        weight_decay=0.01,
    )
    return model, optimizer


def make_loader(options, rows, class_count, epoch=None):
    dataset = AnimalActClips(options.dataset, rows, class_count, options.frames, options.stride,
                            options.image_size)
    rng = torch.Generator().manual_seed(options.seed + (epoch or 0))
    return DataLoader(dataset, batch_size=options.batch_size, shuffle=epoch is not None,
                      num_workers=options.workers, pin_memory=True, generator=rng)


def limit_per_class(rows, class_count, count, seed):
    rng = random.Random(seed)
    selected = []
    for label in range(class_count):
        selected.extend(rng.sample([row for row in rows if row["labels"] == [label]], count))
    return selected


def main():
    options = args()
    if not torch.cuda.is_available():
        raise RuntimeError("This project requires the configured local CUDA environment.")
    torch.manual_seed(options.seed)
    torch.set_num_threads(2)
    random.seed(options.seed)
    np.random.seed(options.seed)
    torch.backends.cudnn.benchmark = False
    manifest = read_json(options.manifest)
    classes = manifest["classes"]
    if options.smoke != (manifest["mode"] == "smoke"):
        raise ValueError("Run mode must match the training manifest mode.")
    train_rows = manifest["splits"]["train"]
    valid_rows = manifest["splits"]["valid"]
    if options.smoke:
        train_rows = limit_per_class(train_rows, len(classes), 4, options.seed)
        valid_rows = limit_per_class(valid_rows, len(classes), 3, options.seed)
    settings = {key: str(value.resolve()) if isinstance(value, Path) else value
                for key, value in vars(options).items() if key not in {"resume", "epochs", "test"}}
    settings["classes"] = classes
    settings["dataset_revision"] = manifest["dataset_revision"]
    settings["split_version"] = manifest["split_version"]
    settings["short_clip_max_seconds"] = SHORT_SECONDS
    out = options.out
    if options.resume:
        if read_json(out / "settings.json") != settings:
            raise ValueError("Resume requires the original settings and dataset.")
    else:
        out.mkdir(parents=True, exist_ok=False)
        write_json(out / "settings.json", settings)

    record_run(out, options)
    model, optimizer = make_model(options, len(classes))
    scaler = torch.amp.GradScaler("cuda")
    tracked_weights = {}
    if options.smoke:
        tracked_weights = {
            "classifier": model.cls_head.fc_cls.weight,
            "slow_layer4": next(model.backbone.slow_path.layer4.parameters()),
            "fast_layer4": next(model.backbone.fast_path.layer4.parameters()),
        }
    before = {name: weight.detach().clone() for name, weight in tracked_weights.items()}
    mean = torch.tensor([123.675, 116.28, 103.53], device="cuda").view(1, 3, 1, 1, 1)
    std = torch.tensor([58.395, 57.12, 57.375], device="cuda").view(1, 3, 1, 1, 1)
    history = []
    best_map = -1.0
    first_epoch = 1
    if options.resume:
        checkpoint = torch.load(out / "last.pth", map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        history = checkpoint["history"]
        best_map = checkpoint["best_map"]
        first_epoch = checkpoint["epoch"] + 1

    for epoch in range(first_epoch, options.epochs + 1):
        torch.manual_seed(options.seed + epoch)
        train_loader = make_loader(options, train_rows, len(classes), epoch=epoch)
        loss = train_epoch(model, train_loader, optimizer, scaler, mean, std, epoch, options.log_every)
        valid_loader = make_loader(options, valid_rows, len(classes))
        valid = predict(model, valid_loader, mean, std)
        threshold = select_threshold(*valid)
        report = evaluate(*valid, classes, threshold)
        row = {"epoch": epoch, "train_loss": loss, "valid_mAP": report["mAP"],
               "valid_macro_F1": report["macro_F1"], "valid_threshold": threshold,
               "gpu_peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 1024**2)}
        history.append(row)
        print(row, flush=True)
        if report["mAP"] is not None and report["mAP"] > best_map:
            best_map = report["mAP"]
            save_checkpoint(out / "best.pth", {"state_dict": model.state_dict(), "classes": classes,
                                               "epoch": epoch, "threshold": threshold})
            save_split_report(out, "valid", valid, classes, threshold, valid_rows)
        save_checkpoint(out / "last.pth", {"state_dict": model.state_dict(),
                                          "optimizer": optimizer.state_dict(),
                                          "scaler": scaler.state_dict(), "epoch": epoch,
                                          "best_map": best_map, "history": history})
        write_json(out / "history.json", history)

    if options.smoke:
        updates = {name: float((parameter.detach() - before[name]).abs().max())
                   for name, parameter in tracked_weights.items()}
        if any(delta <= 0 for delta in updates.values()):
            raise RuntimeError(f"Smoke training failed to update all trainable components: {updates}")
        write_json(out / "weight_updates.json", updates)
        print(f"Smoke weight updates verified: {updates}", flush=True)
    if options.test:
        best = torch.load(out / "best.pth", map_location="cpu", weights_only=True)
        model.load_state_dict(best["state_dict"])
        test_rows = manifest["splits"]["test"]
        test = predict(model, make_loader(options, test_rows, len(classes)), mean, std)
        result = save_split_report(out, "test", test, classes, best["threshold"], test_rows)
        print(f"test  mAP={result['mAP']:.4f}  Macro-F1={result['macro_F1']:.4f}  "
              f"evaluated {result['ap_evaluable_classes']} classes", flush=True)
    print(f"Artifacts: {out.resolve()}", flush=True)


if __name__ == "__main__":
    main()

