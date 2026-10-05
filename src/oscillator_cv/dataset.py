from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset

from oscillator_cv.geometry import (
    board_rotation_degrees,
    potentiometer_angle_degrees,
    sin_cos_degrees,
)
from oscillator_cv.imaging import pil_to_tensor_array
from oscillator_cv.manifest import read_annotations, read_dataset_manifest


def gaussian_heatmap(
    width: int, height: int, point: tuple[float, float], sigma: float = 2.0
) -> torch.Tensor:
    y = torch.arange(height, dtype=torch.float32).view(height, 1)
    x = torch.arange(width, dtype=torch.float32).view(1, width)
    px, py = point
    return torch.exp(-((x - px) ** 2 + (y - py) ** 2) / (2.0 * sigma**2))


def _target_dict(
    width: int,
    height: int,
    axis_tail: tuple[float, float],
    axis_head: tuple[float, float],
    center: tuple[float, float],
    tip: tuple[float, float],
    switch_states: tuple[int, int],
) -> dict[str, torch.Tensor]:
    rotation = board_rotation_degrees(axis_tail, axis_head)
    pot_angle = potentiometer_angle_degrees(center, tip, rotation)
    sin_value, cos_value = sin_cos_degrees(rotation)
    heatmap_size = (width // 2, height // 2)
    center_half = (center[0] / 2.0, center[1] / 2.0)
    tip_half = (tip[0] / 2.0, tip[1] / 2.0)
    heatmaps = torch.stack(
        (
            gaussian_heatmap(heatmap_size[0], heatmap_size[1], center_half),
            gaussian_heatmap(heatmap_size[0], heatmap_size[1], tip_half),
        )
    )
    return {
        "orientation_sin_cos": torch.tensor([sin_value, cos_value], dtype=torch.float32),
        "switch_states": torch.tensor(switch_states, dtype=torch.float32),
        "heatmaps": heatmaps,
        "points": torch.tensor([center, tip], dtype=torch.float32),
        "board_rotation_deg": torch.tensor(rotation, dtype=torch.float32),
        "pot_angle_deg": torch.tensor(pot_angle, dtype=torch.float32),
    }


class AnnotatedImageDataset(Dataset[tuple[torch.Tensor, dict[str, torch.Tensor], str]]):
    def __init__(self, manifest_path: Path, annotations_path: Path, split: str | None = None) -> None:
        payload = read_dataset_manifest(manifest_path)
        annotations = read_annotations(annotations_path)
        records = []
        for record in payload["images"]:
            annotation = annotations.get(record["image_id"])
            if annotation is None or annotation["status"] != "valid":
                continue
            if split is not None and record.get("split") != split:
                continue
            records.append((record, annotation))
        if not records:
            raise ValueError("no valid annotated images matched the requested split")
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor], str]:
        record, annotation = self.records[index]
        with Image.open(record["working_path"]) as source:
            image = source.convert("RGB")
        image_tensor = torch.from_numpy(pil_to_tensor_array(image).copy())
        target = _target_dict(
            image.width,
            image.height,
            tuple(annotation["axis_tail"]),
            tuple(annotation["axis_head"]),
            tuple(annotation["pot_center"]),
            tuple(annotation["pot_tip"]),
            (int(annotation["s_left_on"]), int(annotation["s_right_on"])),
        )
        return image_tensor, target, record["image_id"]


class SyntheticOscillatorDataset(Dataset[tuple[torch.Tensor, dict[str, torch.Tensor], str]]):
    """Deterministic geometric fixtures for pipeline smoke tests, never model evidence."""

    def __init__(self, samples: int = 16, width: int = 96, height: int = 128, seed: int = 7) -> None:
        if width % 16 or height % 16:
            raise ValueError("synthetic width and height must be divisible by 16")
        self.samples = samples
        self.width = width
        self.height = height
        self.seed = seed

    def __len__(self) -> int:
        return self.samples

    @staticmethod
    def _rotate(point: tuple[float, float], center: tuple[float, float], degrees: float) -> tuple[float, float]:
        radians = math.radians(-degrees)  # image y points downward
        dx, dy = point[0] - center[0], point[1] - center[1]
        return (
            center[0] + dx * math.cos(radians) - dy * math.sin(radians),
            center[1] + dx * math.sin(radians) + dy * math.cos(radians),
        )

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor], str]:
        rng = random.Random(self.seed + index)
        rotation = rng.uniform(-25.0, 25.0)
        pot_angle = rng.choice(tuple(range(45, 316, 30)))
        switches = (index % 2, (index // 2) % 2)
        board_center = (self.width / 2.0, self.height / 2.0)
        axis_tail = self._rotate((self.width / 2.0, self.height * 0.82), board_center, rotation)
        axis_head = self._rotate((self.width / 2.0, self.height * 0.18), board_center, rotation)
        center = self._rotate((self.width * 0.42, self.height * 0.52), board_center, rotation)
        pointer_radius = min(self.width, self.height) * 0.10
        pointer_image_angle = math.radians(pot_angle + rotation)
        tip = (
            center[0] + pointer_radius * math.cos(pointer_image_angle),
            center[1] - pointer_radius * math.sin(pointer_image_angle),
        )

        image = Image.new("RGB", (self.width, self.height), (35, 35, 35))
        draw = ImageDraw.Draw(image)
        board_box = [self.width * 0.16, self.height * 0.12, self.width * 0.84, self.height * 0.88]
        draw.rectangle(board_box, fill=(225, 225, 215), outline=(80, 130, 190), width=2)
        draw.line((axis_tail, axis_head), fill=(220, 45, 45), width=max(1, self.width // 48))
        radius = max(3, self.width // 12)
        draw.ellipse((center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius), fill=(40, 90, 190))
        draw.line((center, tip), fill=(245, 130, 25), width=max(2, self.width // 32))
        for switch_index, state in enumerate(switches):
            x = self.width * (0.58 + switch_index * 0.14)
            y = self.height * 0.56
            color = (35, 110, 225) if state else (25, 55, 105)
            draw.rectangle((x - 4, y - 8, x + 4, y + 8), fill=color)
        array = pil_to_tensor_array(image)
        noise = np.random.default_rng(self.seed + index).normal(0.0, 0.01, array.shape).astype(np.float32)
        image_tensor = torch.from_numpy(np.clip(array + noise, 0.0, 1.0))
        target = _target_dict(self.width, self.height, axis_tail, axis_head, center, tip, switches)
        return image_tensor, target, f"synthetic-{index:04d}"
