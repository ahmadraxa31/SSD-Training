from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset
from torchvision.models.detection import ssdlite320_mobilenet_v3_large

from coco_det_dataset import YoloDetectionDataset, detection_collate_fn
from scripts.metrics_det import evaluate_detection_model
from transforms_det import get_eval_transforms, get_train_transforms


@dataclass
class TrainConfig:
    dataset_root: str = r"C:\Users\DF\Documents\YOLO object detection\dataset_full6"
    epochs: int = 5
    batch_size: int = 16
    img_size: int = 320
    lr: float = 0.002
    weight_decay: float = 5e-4
    num_workers: int = 4
    device: str = "0"
    project: str = "ssd-course-project"
    name: str = "ssd-full6-exp1"
    out_root: str = "runs"
    seed: int = 42
    limit_train: int = 0
    limit_val: int = 0
    log_interval: int = 50
    use_wandb: bool = True
    from_scratch: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SSD on YOLO-format detection dataset")
    parser.add_argument("--dataset-root", type=str, default=TrainConfig.dataset_root)
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--img-size", type=int, default=TrainConfig.img_size)
    parser.add_argument("--lr", type=float, default=TrainConfig.lr)
    parser.add_argument("--weight-decay", type=float, default=TrainConfig.weight_decay)
    parser.add_argument("--num-workers", type=int, default=TrainConfig.num_workers)
    parser.add_argument("--device", type=str, default=TrainConfig.device)
    parser.add_argument("--project", type=str, default=TrainConfig.project)
    parser.add_argument("--name", type=str, default=TrainConfig.name)
    parser.add_argument("--out-root", type=str, default=TrainConfig.out_root)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument("--limit-train", type=int, default=TrainConfig.limit_train)
    parser.add_argument("--limit-val", type=int, default=TrainConfig.limit_val)
    parser.add_argument("--log-interval", type=int, default=TrainConfig.log_interval)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--from-scratch", action="store_true", help="Train SSD with random init (no pretrained weights)")
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg.lower() == "cpu" or not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(f"cuda:{device_arg}")


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def maybe_subset(dataset, limit: int):
    if limit <= 0 or limit >= len(dataset):
        return dataset
    return Subset(dataset, list(range(limit)))


def maybe_init_wandb(cfg: TrainConfig):
    if not cfg.use_wandb:
        return None
    try:
        import wandb

        run = wandb.init(project=cfg.project, name=cfg.name, config=asdict(cfg))
        return run
    except Exception as exc:
        print(f"[WARN] W&B disabled due to init failure: {exc}")
        return None


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    log_interval: int,
) -> dict[str, float]:
    model.train()
    running = {"loss": 0.0, "bbox_regression": 0.0, "classification": 0.0}
    n_steps = 0

    for step, (images, targets) in enumerate(dataloader, start=1):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        running["loss"] += float(loss.item())
        running["bbox_regression"] += float(loss_dict["bbox_regression"].item())
        running["classification"] += float(loss_dict["classification"].item())
        n_steps += 1

        if step % log_interval == 0:
            print(
                f"step {step}/{len(dataloader)} "
                f"loss={running['loss']/n_steps:.4f} "
                f"bbox={running['bbox_regression']/n_steps:.4f} "
                f"cls={running['classification']/n_steps:.4f}"
            )

    for key in running:
        running[key] /= max(n_steps, 1)
    return running


def main() -> None:
    args = parse_args()
    cfg = TrainConfig(
        dataset_root=args.dataset_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        img_size=args.img_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        device=args.device,
        project=args.project,
        name=args.name,
        out_root=args.out_root,
        seed=args.seed,
        limit_train=args.limit_train,
        limit_val=args.limit_val,
        log_interval=args.log_interval,
        use_wandb=not args.no_wandb,
        from_scratch=args.from_scratch,
    )

    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    print(f"Using device: {device}")

    train_ds = YoloDetectionDataset(
        dataset_root=cfg.dataset_root,
        split="train",
        transforms=get_train_transforms(img_size=cfg.img_size),
    )
    val_ds = YoloDetectionDataset(
        dataset_root=cfg.dataset_root,
        split="val",
        transforms=get_eval_transforms(img_size=cfg.img_size),
    )
    train_ds = maybe_subset(train_ds, cfg.limit_train)
    val_ds = maybe_subset(val_ds, cfg.limit_val)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=detection_collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=detection_collate_fn,
    )

    # +1 for background class index 0.
    num_classes = len(train_ds.dataset.class_names) + 1 if isinstance(train_ds, Subset) else len(train_ds.class_names) + 1
    class_names = train_ds.dataset.class_names if isinstance(train_ds, Subset) else train_ds.class_names
    if cfg.from_scratch:
        model = ssdlite320_mobilenet_v3_large(
            weights=None,
            weights_backbone=None,
            num_classes=num_classes,
        )
    else:
        model = ssdlite320_mobilenet_v3_large(
            weights=None,
            weights_backbone="DEFAULT",
            num_classes=num_classes,
        )
    model.to(device)

    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.lr,
        momentum=0.9,
        weight_decay=cfg.weight_decay,
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=max(cfg.epochs // 2, 1),
        gamma=0.1,
    )

    out_dir = Path(cfg.out_root) / cfg.project / cfg.name
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "results.csv"
    ckpt_last = out_dir / "last.pth"
    ckpt_best = out_dir / "best.pth"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "train_loss",
                "train_bbox",
                "train_cls",
                "val_map50_95",
                "val_map50",
                "val_mar100",
                "val_precision",
                "val_recall",
                "lr",
            ],
        )
        writer.writeheader()

    wb_run = maybe_init_wandb(cfg)

    best_map = -1.0
    for epoch in range(1, cfg.epochs + 1):
        print(f"\nEpoch {epoch}/{cfg.epochs}")
        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            log_interval=cfg.log_interval,
        )
        val_metrics = evaluate_detection_model(
            model=model,
            dataloader=val_loader,
            device=device,
        )
        lr_now = optimizer.param_groups[0]["lr"]
        lr_scheduler.step()

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_bbox": train_metrics["bbox_regression"],
            "train_cls": train_metrics["classification"],
            "val_map50_95": val_metrics["map50_95"],
            "val_map50": val_metrics["map50"],
            "val_mar100": val_metrics["mar100"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "lr": lr_now,
        }
        with csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            writer.writerow(row)

        print(
            f"train_loss={row['train_loss']:.4f} "
            f"map50_95={row['val_map50_95']:.4f} map50={row['val_map50']:.4f} "
            f"precision={row['val_precision']:.4f} recall={row['val_recall']:.4f}"
        )

        checkpoint: dict[str, Any] = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "class_names": class_names,
            "num_classes": num_classes,
            "img_size": cfg.img_size,
            "best_map50_95": max(best_map, row["val_map50_95"]),
            "config": asdict(cfg),
        }
        torch.save(checkpoint, ckpt_last)
        if row["val_map50_95"] > best_map:
            best_map = row["val_map50_95"]
            torch.save(checkpoint, ckpt_best)

        if wb_run is not None:
            import wandb

            wandb.log(row, step=epoch)

    if wb_run is not None:
        wb_run.finish()

    print(f"\nDone. Best val mAP50-95: {best_map:.4f}")
    print(f"Saved to: {out_dir}")


if __name__ == "__main__":
    main()
