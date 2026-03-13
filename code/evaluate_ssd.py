from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import ssdlite320_mobilenet_v3_large

from coco_det_dataset import YoloDetectionDataset, detection_collate_fn
from scripts.metrics_det import evaluate_detection_model
from transforms_det import get_eval_transforms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SSD checkpoint")
    parser.add_argument("--weights", type=str, required=True, help="Path to .pth checkpoint")
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=r"C:\Users\DF\Documents\YOLO object detection\dataset_full6",
    )
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--img-size", type=int, default=320)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--out", type=str, default="runs/eval_ssd")
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg.lower() == "cpu" or not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(f"cuda:{device_arg}")


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    ckpt = torch.load(args.weights, map_location="cpu")

    class_names = ckpt.get("class_names")
    if class_names is None:
        raise KeyError("Checkpoint must contain `class_names`")
    num_classes = int(ckpt.get("num_classes", len(class_names) + 1))
    img_size = int(ckpt.get("img_size", args.img_size))

    model = ssdlite320_mobilenet_v3_large(
        weights=None,
        weights_backbone=None,
        num_classes=num_classes,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    dataset = YoloDetectionDataset(
        dataset_root=args.dataset_root,
        split=args.split,
        transforms=get_eval_transforms(img_size=img_size),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=detection_collate_fn,
    )

    metrics = evaluate_detection_model(model=model, dataloader=loader, device=device)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"metrics_{args.split}.csv"
    pd.DataFrame([metrics]).to_csv(out_file, index=False)
    print(pd.DataFrame([metrics]).to_string(index=False))
    print(f"Saved metrics to: {out_file}")


if __name__ == "__main__":
    main()
