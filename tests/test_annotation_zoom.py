from PIL import Image
import pytest

from oscillator_cv.annotation_zoom import PotZoomView
from oscillator_cv.geometry import make_letterbox_transform


def test_zoom_preserves_fractional_working_coordinates() -> None:
    original = Image.new("RGB", (300, 400), "blue")
    transform = make_letterbox_transform(300, 400, 60, 80)
    zoom = PotZoomView.create(original, transform, (30, 35), radius=8)
    point = (30.125, 36.375)
    displayed = zoom.working_to_display_point(point)
    assert zoom.display_to_working_point(displayed) == pytest.approx(point)
    assert zoom.display_to_working_point((0, 0)) is None
    # Selecting the crop's displayed center maps to the original working center.
    assert zoom.display_to_working_point((288, 384)) == pytest.approx((30, 35))


def test_zoom_near_image_edge_keeps_correct_mapping() -> None:
    original = Image.new("RGB", (300, 400), "blue")
    transform = make_letterbox_transform(300, 400, 60, 80)
    zoom = PotZoomView.create(original, transform, (1, 2), radius=8)
    point = (1.125, 2.625)
    assert zoom.display_to_working_point(zoom.working_to_display_point(point)) == pytest.approx(point)


def test_zoom_rejects_stale_landscape_manifest() -> None:
    original = Image.new("RGB", (300, 400), "blue")
    stale_transform = make_letterbox_transform(400, 300)
    with pytest.raises(ValueError, match="orientation differs"):
        PotZoomView.create(original, stale_transform, (200, 300))


def test_pot_review_preserves_other_labels_and_resumes(monkeypatch, tmp_path) -> None:
    import numpy as np
    import oscillator_cv.annotation as annotation_module

    transform = make_letterbox_transform(1200, 1600)
    record = {"image_id": "pilot", "working_path": "unused.jpg", "source_path": "original.jpg",
              "transform": transform.to_dict()}
    previous = {"image_id": "pilot", "status": "valid", "s_left_on": 1, "s_right_on": 0,
                "axis_tail": [100, 650], "axis_head": [100, 150],
                "pot_center": [280, 300], "pot_tip": [285, 310], "notes": "preserve me"}
    saved = {"pilot": previous}
    callbacks = {}
    monkeypatch.setattr(annotation_module, "read_dataset_manifest", lambda _: {"images": [record]})
    monkeypatch.setattr(annotation_module, "read_annotations", lambda _: saved)
    monkeypatch.setattr(annotation_module, "write_annotations", lambda *_: None)
    monkeypatch.setattr(annotation_module, "load_board_portrait_rgb", lambda _: Image.new("RGB", (1200, 1600)))
    monkeypatch.setattr(annotation_module.cv2, "imread", lambda *_: np.zeros((768, 576, 3), dtype=np.uint8))
    for name in ("namedWindow", "resizeWindow", "imshow", "destroyWindow", "destroyAllWindows"):
        monkeypatch.setattr(annotation_module.cv2, name, lambda *_: None)
    monkeypatch.setattr(annotation_module.cv2, "setMouseCallback", lambda _, callback: callbacks.update(mouse=callback))

    def click_then_save(_):
        callbacks["mouse"](annotation_module.cv2.EVENT_LBUTTONDOWN, 290, 385, 0, None)
        callbacks["mouse"](annotation_module.cv2.EVENT_LBUTTONDOWN, 350, 430, 0, None)
        return ord("n")

    monkeypatch.setattr(annotation_module.cv2, "waitKey", click_then_save)
    result = annotation_module.annotate_interactive(tmp_path / "manifest.json", tmp_path / "labels.json", review_pot=True)
    assert result["annotations_written"] == 1
    for key in ("axis_tail", "axis_head", "s_left_on", "s_right_on", "notes"):
        assert saved["pilot"][key] == previous[key]
    assert saved["pilot"]["pot_center"] != previous["pot_center"]
    assert saved["pilot"]["pot_annotation_method"] == "original-resolution-zoom-v1"
    # A second review skips the completed image rather than asking for its points again.
    assert annotation_module.annotate_interactive(tmp_path / "manifest.json", tmp_path / "labels.json", review_pot=True)["annotations_written"] == 0
