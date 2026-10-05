from __future__ import annotations

import math
from dataclasses import dataclass

from PIL import Image

from oscillator_cv.geometry import LetterboxTransform
from oscillator_cv.imaging import letterbox_image


@dataclass
class PotZoomView:
    image: Image.Image
    crop_origin: tuple[int, int]
    crop_size: tuple[int, int]
    display_transform: LetterboxTransform
    working_transform: LetterboxTransform

    @classmethod
    def create(cls, original: Image.Image, working_transform: LetterboxTransform,
               working_center: tuple[float, float], radius: float = 40) -> PotZoomView:
        if original.size != (working_transform.source_width, working_transform.source_height):
            raise ValueError("Image orientation differs from manifest; reimport with current portrait preprocessing.")
        cx, cy = working_transform.inverse_point(working_center)
        distance = radius / working_transform.scale
        x0 = max(0, math.floor(cx - distance))
        y0 = max(0, math.floor(cy - distance))
        x1 = min(original.width, math.ceil(cx + distance))
        y1 = min(original.height, math.ceil(cy + distance))
        if x1 <= x0 or y1 <= y0:
            raise ValueError("Potentiometer crop is outside the photograph.")
        crop = original.crop((x0, y0, x1, y1))
        display, transform = letterbox_image(crop, 576, 768)
        return cls(display, (x0, y0), crop.size, transform, working_transform)

    def display_to_working_point(self, point: tuple[float, float]) -> tuple[float, float] | None:
        x, y = self.display_transform.inverse_point(point)
        if not (0 <= x < self.crop_size[0] and 0 <= y < self.crop_size[1]):
            return None
        return self.working_transform.forward_point((x + self.crop_origin[0], y + self.crop_origin[1]))

    def working_to_display_point(self, point: tuple[float, float]) -> tuple[float, float]:
        x, y = self.working_transform.inverse_point(point)
        return self.display_transform.forward_point((x - self.crop_origin[0], y - self.crop_origin[1]))
