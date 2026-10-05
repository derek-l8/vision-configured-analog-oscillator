from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from oscillator_cv.geometry import LetterboxTransform, make_letterbox_transform


WORKING_WIDTH = 576
WORKING_HEIGHT = 768


def load_exif_normalized_rgb(path: Path) -> Image.Image:
    """Load a JPEG without changing it on disk and apply its EXIF orientation."""
    with Image.open(path) as source:
        return ImageOps.exif_transpose(source).convert("RGB")


def load_board_portrait_rgb(path: Path) -> Image.Image:
    """Apply EXIF, then turn landscape input clockwise before model resizing.

    Capture convention: power module at the portrait top, or at the landscape
    left. This quarter-turn does not estimate or remove small board rotation.
    Model/annotation coordinates refer to this canonical portrait image.
    """
    image = load_exif_normalized_rgb(path)
    if image.width > image.height:
        return image.transpose(Image.Transpose.ROTATE_270)
    return image


def letterbox_image(
    image: Image.Image, width: int = WORKING_WIDTH, height: int = WORKING_HEIGHT
) -> tuple[Image.Image, LetterboxTransform]:
    transform = make_letterbox_transform(image.width, image.height, width, height)
    resized = image.resize((transform.resized_width, transform.resized_height), Image.Resampling.LANCZOS)
    output = Image.new("RGB", (width, height), (24, 24, 24))
    output.paste(resized, (transform.offset_x, transform.offset_y))
    return output, transform


def normalize_image_file(
    source: Path, destination: Path, width: int = WORKING_WIDTH, height: int = WORKING_HEIGHT
) -> tuple[LetterboxTransform, tuple[int, int]]:
    image = load_board_portrait_rgb(source)
    original_size = image.size
    normalized, transform = letterbox_image(image, width, height)
    destination.parent.mkdir(parents=True, exist_ok=True)
    normalized.save(destination, format="JPEG", quality=95, subsampling=0)
    return transform, original_size


def pil_to_tensor_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
