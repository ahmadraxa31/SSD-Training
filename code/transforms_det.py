from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable

import torch
from torchvision.transforms import functional as F


class Compose:
    def __init__(self, transforms: list[Callable]):
        self.transforms = transforms

    def __call__(self, image, target):
        for transform in self.transforms:
            image, target = transform(image, target)
        return image, target


class ToTensor:
    def __call__(self, image, target):
        return F.to_tensor(image), target


@dataclass
class Resize:
    width: int
    height: int

    def __call__(self, image, target):
        old_w, old_h = image.size
        image = F.resize(image, [self.height, self.width])

        boxes = target["boxes"]
        if boxes.numel() > 0:
            scale_x = self.width / old_w
            scale_y = self.height / old_h
            boxes = boxes.clone()
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale_x
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale_y
            target["boxes"] = boxes
            target["area"] = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        return image, target


@dataclass
class RandomHorizontalFlip:
    p: float = 0.5

    def __call__(self, image, target):
        if random.random() >= self.p:
            return image, target

        width, _ = image.size
        image = F.hflip(image)

        boxes = target["boxes"]
        if boxes.numel() > 0:
            boxes = boxes.clone()
            x1 = boxes[:, 0].clone()
            x2 = boxes[:, 2].clone()
            boxes[:, 0] = width - x2
            boxes[:, 2] = width - x1
            target["boxes"] = boxes

        return image, target


def get_train_transforms(img_size: int = 320) -> Compose:
    return Compose(
        [
            Resize(width=img_size, height=img_size),
            RandomHorizontalFlip(p=0.5),
            ToTensor(),
        ]
    )


def get_eval_transforms(img_size: int = 320) -> Compose:
    return Compose(
        [
            Resize(width=img_size, height=img_size),
            ToTensor(),
        ]
    )
