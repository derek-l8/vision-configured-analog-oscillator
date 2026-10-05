from pathlib import Path

from PIL import Image

from oscillator_cv.imaging import load_board_portrait_rgb, load_exif_normalized_rgb, normalize_image_file


def test_exif_orientation_is_applied_without_modifying_original(tmp_path: Path) -> None:
    source = tmp_path / "rotated.jpg"
    destination = tmp_path / "processed" / "rotated.jpg"
    image = Image.new("RGB", (20, 10), (200, 20, 10))
    exif = Image.Exif()
    exif[274] = 6
    image.save(source, exif=exif)
    original_bytes = source.read_bytes()

    normalized = load_exif_normalized_rgb(source)
    assert normalized.size == (10, 20)
    transform, source_size = normalize_image_file(source, destination)

    assert source.read_bytes() == original_bytes
    assert source_size == (10, 20)
    assert transform.source_width == 10
    with Image.open(destination) as working:
        assert working.mode == "RGB"
        assert working.size == (576, 768)
        assert working.getexif().get(274) is None


def test_landscape_and_exif_portrait_have_same_model_input(tmp_path: Path) -> None:
    image = Image.new("RGB", (40, 20), (10, 20, 180))
    image.paste((230, 30, 10), (0, 0, 12, 20))
    paths = []
    original_bytes = []
    for orientation in (1, 6):
        path = tmp_path / f"orientation-{orientation}.jpg"
        exif = Image.Exif()
        exif[274] = orientation
        image.save(path, exif=exif, quality=95, subsampling=0)
        paths.append(path)
        original_bytes.append(path.read_bytes())

    landscape = load_board_portrait_rgb(paths[0])
    exif_portrait = load_board_portrait_rgb(paths[1])
    assert landscape.size == exif_portrait.size == (20, 40)
    assert landscape.tobytes() == exif_portrait.tobytes()
    assert landscape.getpixel((10, 3))[0] > 200  # Landscape left becomes top.
    for path, before in zip(paths, original_bytes):
        transform, source_size = normalize_image_file(path, tmp_path / "processed" / path.name)
        assert source_size == (20, 40)
        assert transform.resized_height == 768
        assert transform.offset_y == 0
        assert path.read_bytes() == before


def test_portrait_input_is_not_rotated_again(tmp_path: Path) -> None:
    path = tmp_path / "portrait.jpg"
    image = Image.new("RGB", (20, 40), (10, 20, 180))
    image.paste((230, 30, 10), (0, 0, 20, 12))
    image.save(path, quality=95, subsampling=0)
    exif_only = load_exif_normalized_rgb(path)
    canonical = load_board_portrait_rgb(path)
    assert canonical.size == (20, 40)
    assert canonical.tobytes() == exif_only.tobytes()
