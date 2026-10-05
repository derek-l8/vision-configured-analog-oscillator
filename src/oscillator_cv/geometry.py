from __future__ import annotations

import math
from dataclasses import dataclass


def wrap_degrees(angle: float) -> float:
    """Wrap an angle to [-180, 180)."""
    return (angle + 180.0) % 360.0 - 180.0


def wrap_degrees_360(angle: float) -> float:
    """Wrap an angle to [0, 360)."""
    return angle % 360.0


def circular_error_degrees(predicted: float, target: float) -> float:
    return abs(wrap_degrees(predicted - target))


def vector_to_math_angle_degrees(origin: tuple[float, float], tip: tuple[float, float]) -> float:
    """Return an angle with +x right, +y up, positive counterclockwise."""
    dx = tip[0] - origin[0]
    dy = origin[1] - tip[1]
    if dx == 0.0 and dy == 0.0:
        raise ValueError("angle points must be distinct")
    return math.degrees(math.atan2(dy, dx))


def board_rotation_degrees(axis_tail: tuple[float, float], axis_head: tuple[float, float]) -> float:
    """Board rotation relative to canonical image-up.

    The ordered axis runs from the buzzer/lower end toward the power-module/upper
    end. A canonical upright board therefore has rotation 0 degrees.
    """
    axis_angle = vector_to_math_angle_degrees(axis_tail, axis_head)
    return wrap_degrees(axis_angle - 90.0)


def potentiometer_angle_degrees(
    center: tuple[float, float], tip: tuple[float, float], board_rotation: float
) -> float:
    image_angle = vector_to_math_angle_degrees(center, tip)
    return wrap_degrees_360(image_angle - board_rotation)


def sin_cos_degrees(angle: float) -> tuple[float, float]:
    radians = math.radians(angle)
    return math.sin(radians), math.cos(radians)


def degrees_from_sin_cos(sin_value: float, cos_value: float) -> float:
    return wrap_degrees(math.degrees(math.atan2(sin_value, cos_value)))


@dataclass(frozen=True)
class LetterboxTransform:
    source_width: int
    source_height: int
    target_width: int
    target_height: int
    scale: float
    offset_x: int
    offset_y: int
    resized_width: int
    resized_height: int

    def forward_point(self, point: tuple[float, float]) -> tuple[float, float]:
        return point[0] * self.scale + self.offset_x, point[1] * self.scale + self.offset_y

    def inverse_point(self, point: tuple[float, float]) -> tuple[float, float]:
        return (point[0] - self.offset_x) / self.scale, (point[1] - self.offset_y) / self.scale

    def to_dict(self) -> dict[str, int | float]:
        return {
            "source_width": self.source_width,
            "source_height": self.source_height,
            "target_width": self.target_width,
            "target_height": self.target_height,
            "scale": self.scale,
            "offset_x": self.offset_x,
            "offset_y": self.offset_y,
            "resized_width": self.resized_width,
            "resized_height": self.resized_height,
        }


def make_letterbox_transform(
    source_width: int, source_height: int, target_width: int = 576, target_height: int = 768
) -> LetterboxTransform:
    if min(source_width, source_height, target_width, target_height) <= 0:
        raise ValueError("image dimensions must be positive")
    scale = min(target_width / source_width, target_height / source_height)
    resized_width = max(1, round(source_width * scale))
    resized_height = max(1, round(source_height * scale))
    offset_x = (target_width - resized_width) // 2
    offset_y = (target_height - resized_height) // 2
    return LetterboxTransform(
        source_width,
        source_height,
        target_width,
        target_height,
        scale,
        offset_x,
        offset_y,
        resized_width,
        resized_height,
    )
