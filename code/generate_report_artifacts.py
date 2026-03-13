from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd
import torch
from PIL import Image, ImageDraw
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.transforms.functional import pil_to_tensor


def load_yolo_boxes(label_path: Path, width: int, height: int):
    boxes = []
    labels = []
    if not label_path.exists():
        return boxes, labels
    text = label_path.read_text(encoding='utf-8').strip()
    if not text:
        return boxes, labels
    for line in text.splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        cid, xc, yc, w, h = parts
        cid = int(float(cid))
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
        boxes.append((x1, y1, x2, y2))
        labels.append(cid)
    return boxes, labels


def draw_boxes(image: Image.Image, boxes: Iterable[tuple[float, float, float, float]], labels: Iterable[str], color: str):
    draw = ImageDraw.Draw(image)
    for box, label in zip(boxes, labels):
        x1, y1, x2, y2 = box
        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
        text_xy = (x1 + 3, max(0, y1 - 14))
        draw.text(text_xy, label, fill=color)


def make_training_curves(results_csv: Path, out_path: Path):
    df = pd.read_csv(results_csv)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(df['epoch'], df['train_loss'], marker='o', linewidth=2, label='Train Loss')
    axes[0].set_title('Training Loss by Epoch')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].grid(alpha=0.3)

    axes[1].plot(df['epoch'], df['val_map50_95'], marker='o', linewidth=2, label='mAP@0.5:0.95')
    axes[1].plot(df['epoch'], df['val_map50'], marker='o', linewidth=2, label='mAP@0.5')
    axes[1].plot(df['epoch'], df['val_precision'], marker='o', linewidth=2, label='Precision@0.5')
    axes[1].plot(df['epoch'], df['val_recall'], marker='o', linewidth=2, label='Recall@0.5')
    axes[1].set_title('Validation Metrics by Epoch')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Score')
    axes[1].set_ylim(0, 1.0)
    axes[1].grid(alpha=0.3)
    axes[1].legend(loc='lower right', fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def make_class_distribution(out_path: Path):
    classes = ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck']
    train_counts = [257252, 7056, 43533, 8654, 6061, 9970]
    val_counts = [5483, 157, 881, 186, 143, 187]
    test_counts = [5294, 157, 1037, 181, 140, 227]

    x = range(len(classes))
    w = 0.26

    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.bar([i - w for i in x], train_counts, width=w, label='Train')
    ax.bar(x, val_counts, width=w, label='Val')
    ax.bar([i + w for i in x], test_counts, width=w, label='Test')
    ax.set_xticks(list(x))
    ax.set_xticklabels(classes, rotation=0)
    ax.set_ylabel('Number of Objects')
    ax.set_title('Class Distribution Across Splits')
    ax.grid(axis='y', alpha=0.25)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def make_gt_and_prediction_grids(
    ckpt_path: Path,
    data_root: Path,
    out_gt: Path,
    out_pred: Path,
    score_thresh: float = 0.40,
    n_images: int = 4,
):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    class_names = ckpt['class_names']
    num_classes = int(ckpt.get('num_classes', len(class_names) + 1))

    model = ssdlite320_mobilenet_v3_large(weights=None, weights_backbone=None, num_classes=num_classes)
    model.load_state_dict(ckpt['model_state_dict'])

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    test_images = sorted((data_root / 'images' / 'test').glob('*.jpg'))
    if len(test_images) < n_images:
        test_images = sorted((data_root / 'images' / 'test').glob('*'))
    picks = [test_images[i] for i in [0, 25, 120, 350] if i < len(test_images)]
    if len(picks) < n_images:
        picks = test_images[:n_images]

    gt_canvases = []
    pred_canvases = []

    with torch.inference_mode():
        for img_path in picks[:n_images]:
            img = Image.open(img_path).convert('RGB')
            w, h = img.size

            label_path = data_root / 'labels' / 'test' / f'{img_path.stem}.txt'
            gt_boxes, gt_ids = load_yolo_boxes(label_path, w, h)
            gt_labels = [class_names[c] for c in gt_ids]

            gt_img = img.copy()
            draw_boxes(gt_img, gt_boxes, gt_labels, color='lime')
            gt_canvases.append((img_path.name, gt_img))

            x = pil_to_tensor(img).float() / 255.0
            output = model([x.to(device)])[0]
            pboxes = output['boxes'].detach().cpu()
            pscores = output['scores'].detach().cpu()
            plabels = output['labels'].detach().cpu()

            keep = pscores >= score_thresh
            pboxes = pboxes[keep]
            pscores = pscores[keep]
            plabels = plabels[keep]

            pred_img = img.copy()
            labels = []
            box_list = []
            for b, s, l in zip(pboxes[:15], pscores[:15], plabels[:15]):
                cid = int(l.item()) - 1
                if cid < 0 or cid >= len(class_names):
                    continue
                box_list.append(tuple(float(v) for v in b.tolist()))
                labels.append(f"{class_names[cid]} {s.item():.2f}")
            draw_boxes(pred_img, box_list, labels, color='red')
            pred_canvases.append((img_path.name, pred_img))

    def save_grid(items, out_path: Path, title: str):
        fig, axes = plt.subplots(2, 2, figsize=(13, 9))
        axes = axes.flatten()
        for ax, (name, canvas) in zip(axes, items):
            ax.imshow(canvas)
            ax.set_title(name, fontsize=10)
            ax.axis('off')
        for j in range(len(items), 4):
            axes[j].axis('off')
        fig.suptitle(title, fontsize=14)
        fig.tight_layout()
        fig.savefig(out_path, dpi=220)
        plt.close(fig)

    save_grid(gt_canvases, out_gt, 'Ground-Truth Boxes (Test Samples)')
    save_grid(pred_canvases, out_pred, f'SSD Predictions (score >= {score_thresh})')


def main():
    root = Path(r'C:\Users\DF\Documents\SSD-model')
    figures = root / 'report' / 'figures'
    figures.mkdir(parents=True, exist_ok=True)

    results_csv = root / 'runs' / 'ssd-course-project' / 'ssd-full6-scratch-5e-newacct' / 'results.csv'
    ckpt_path = root / 'runs' / 'ssd-course-project' / 'ssd-full6-scratch-5e-newacct' / 'best.pth'
    data_root = Path(r'C:\Users\DF\Documents\YOLO object detection\dataset_full6')

    make_training_curves(results_csv, figures / 'training_curves.png')
    make_class_distribution(figures / 'class_distribution.png')
    make_gt_and_prediction_grids(
        ckpt_path=ckpt_path,
        data_root=data_root,
        out_gt=figures / 'gt_samples.png',
        out_pred=figures / 'pred_samples.png',
        score_thresh=0.40,
        n_images=4,
    )
    print(f'Artifacts saved to: {figures}')


if __name__ == '__main__':
    main()
