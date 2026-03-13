from __future__ import annotations

from pathlib import Path

import torch
import yaml
from PIL import Image
from torch.utils.data import Dataset


def _load_class_names(dataset_root: Path) -> list[str]:
    yaml_path = dataset_root / "dataset.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"Missing dataset.yaml: {yaml_path}")

    with yaml_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    names = data.get("names")
    if isinstance(names, dict):
        ordered = [names[k] for k in sorted(names.keys(), key=lambda x: int(x))]
        return ordered
    if isinstance(names, list):
        return names
    raise ValueError("dataset.yaml must define `names` as list or dict")


class YoloDetectionDataset(Dataset):
    """
    Torchvision detection dataset backed by YOLO-format labels.
    Labels are expected as:
        class_id x_center y_center width height
    where box coordinates are normalized to [0, 1].
    """

    def __init__(
        self,
        dataset_root: str | Path,
        split: str,
        transforms=None,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.split = split
        self.transforms = transforms

        self.images_dir = self.dataset_root / "images" / split
        self.labels_dir = self.dataset_root / "labels" / split
        if not self.images_dir.exists():
            raise FileNotFoundError(f"Missing images directory: {self.images_dir}")
        if not self.labels_dir.exists():
            raise FileNotFoundError(f"Missing labels directory: {self.labels_dir}")

        self.class_names = _load_class_names(self.dataset_root)
        self.image_paths = sorted(
            [
                p
                for p in self.images_dir.iterdir()
                if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
            ]
        )
        if not self.image_paths:
            raise ValueError(f"No images found in {self.images_dir}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def _read_target(self, label_path: Path, width: int, height: int, index: int) -> dict[str, torch.Tensor]:
        boxes = []
        labels = []

        if label_path.exists():
            for line in label_path.read_text(encoding="utf-8").splitlines():
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls_id, xc, yc, w, h = parts
                cls_id = int(float(cls_id))
                xc = float(xc) * width
                yc = float(yc) * height
                bw = float(w) * width
                bh = float(h) * height

                x1 = max(0.0, xc - bw / 2.0)
                y1 = max(0.0, yc - bh / 2.0)
                x2 = min(float(width), xc + bw / 2.0)
                y2 = min(float(height), yc + bh / 2.0)
                if x2 <= x1 or y2 <= y1:
                    continue

                boxes.append([x1, y1, x2, y2])
                # SSD expects class ids starting at 1 (0 is background).
                labels.append(cls_id + 1)

        if boxes:
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
            labels_tensor = torch.tensor(labels, dtype=torch.int64)
            area = (boxes_tensor[:, 2] - boxes_tensor[:, 0]) * (boxes_tensor[:, 3] - boxes_tensor[:, 1])
        else:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)
            area = torch.zeros((0,), dtype=torch.float32)

        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": torch.tensor([index], dtype=torch.int64),
            "area": area,
            "iscrowd": torch.zeros((labels_tensor.shape[0],), dtype=torch.int64),
        }
        return target

    def __getitem__(self, index: int):
        image_path = self.image_paths[index]
        label_path = self.labels_dir / f"{image_path.stem}.txt"

        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        target = self._read_target(label_path=label_path, width=width, height=height, index=index)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target


def detection_collate_fn(batch):
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return images, targets
