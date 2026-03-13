from __future__ import annotations

import argparse
import time
from statistics import mean

import torch
from torchvision.models.detection import ssdlite320_mobilenet_v3_large


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark SSD inference FPS")
    parser.add_argument("--weights", type=str, required=True, help="Path to .pth checkpoint")
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=200)
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg.lower() == "cpu" or not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(f"cuda:{device_arg}")


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    ckpt = torch.load(args.weights, map_location="cpu")

    class_names = ckpt["class_names"]
    num_classes = int(ckpt.get("num_classes", len(class_names) + 1))
    img_size = int(ckpt.get("img_size", 320))

    model = ssdlite320_mobilenet_v3_large(
        weights=None,
        weights_backbone=None,
        num_classes=num_classes,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    x = torch.randn(3, img_size, img_size, device=device)
    with torch.inference_mode():
        for _ in range(args.warmup):
            _ = model([x])
        if device.type == "cuda":
            torch.cuda.synchronize()

        times = []
        for _ in range(args.iters):
            start = time.perf_counter()
            _ = model([x])
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - start)

    avg_latency = mean(times)
    fps = 1.0 / avg_latency
    print(f"Average latency: {avg_latency * 1000:.2f} ms")
    print(f"FPS: {fps:.2f}")


if __name__ == "__main__":
    main()
