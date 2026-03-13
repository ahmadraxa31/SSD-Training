from __future__ import annotations

import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.ops import box_iou


@torch.inference_mode()
def compute_precision_recall_at_iou(
    preds: list[dict[str, torch.Tensor]],
    targets: list[dict[str, torch.Tensor]],
    iou_thresh: float = 0.5,
    score_thresh: float = 0.25,
) -> tuple[float, float]:
    tp = 0
    fp = 0
    fn = 0

    for pred, target in zip(preds, targets):
        pred_boxes = pred["boxes"]
        pred_labels = pred["labels"]
        pred_scores = pred["scores"]
        gt_boxes = target["boxes"]
        gt_labels = target["labels"]

        keep = pred_scores >= score_thresh
        pred_boxes = pred_boxes[keep]
        pred_labels = pred_labels[keep]

        matched_gt = torch.zeros((gt_boxes.shape[0],), dtype=torch.bool, device=gt_boxes.device)

        for pbox, plabel in zip(pred_boxes, pred_labels):
            label_mask = gt_labels == plabel
            candidate_idx = torch.where(label_mask & ~matched_gt)[0]
            if candidate_idx.numel() == 0:
                fp += 1
                continue

            ious = box_iou(pbox.unsqueeze(0), gt_boxes[candidate_idx]).squeeze(0)
            best_iou, best_pos = torch.max(ious, dim=0)
            if float(best_iou) >= iou_thresh:
                tp += 1
                matched_gt[candidate_idx[best_pos]] = True
            else:
                fp += 1

        fn += int((~matched_gt).sum().item())

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return precision, recall


@torch.inference_mode()
def evaluate_detection_model(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    iou_thresh: float = 0.5,
    score_thresh: float = 0.25,
) -> dict[str, float]:
    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")
    model.eval()

    all_preds = []
    all_targets = []

    for images, targets in dataloader:
        images = [img.to(device) for img in images]
        outputs = model(images)

        preds_cpu = [{k: v.detach().cpu() for k, v in out.items()} for out in outputs]
        targets_cpu = [{k: v.detach().cpu() for k, v in tgt.items()} for tgt in targets]

        metric.update(preds_cpu, targets_cpu)
        all_preds.extend(preds_cpu)
        all_targets.extend(targets_cpu)

    m = metric.compute()
    precision, recall = compute_precision_recall_at_iou(
        preds=all_preds,
        targets=all_targets,
        iou_thresh=iou_thresh,
        score_thresh=score_thresh,
    )

    return {
        "map50_95": float(m["map"].item()),
        "map50": float(m["map_50"].item()),
        "mar100": float(m["mar_100"].item()),
        "precision": float(precision),
        "recall": float(recall),
    }
