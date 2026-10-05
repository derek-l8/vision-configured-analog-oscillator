from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from oscillator_cv.imaging import WORKING_HEIGHT, WORKING_WIDTH, normalize_image_file
from oscillator_cv.manifest import (
    read_annotations,
    read_capture_plan,
    read_dataset_manifest,
    validate_capture_plan,
    write_dataset_manifest,
)


JPEG_SUFFIXES = {".jpg", ".jpeg"}


def import_images(
    source_dir: Path,
    output_dir: Path,
    manifest_path: Path,
    capture_plan_path: Path | None = None,
    width: int = WORKING_WIDTH,
    height: int = WORKING_HEIGHT,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {output_dir}")
    sources = sorted(path for path in source_dir.rglob("*") if path.is_file() and path.suffix.lower() in JPEG_SUFFIXES)
    if not sources:
        raise ValueError(f"no JPEG images found under {source_dir}")
    plan_by_id: dict[str, dict[str, Any]] = {}
    if capture_plan_path:
        plan_rows = read_capture_plan(capture_plan_path)
        validate_capture_plan(plan_rows)
        plan_by_id = {str(row["image_id"]): row for row in plan_rows}
    records: list[dict[str, Any]] = []
    used_ids: set[str] = set()
    for index, source in enumerate(sources, start=1):
        image_id = source.stem.lower()
        if image_id in used_ids:
            image_id = f"{image_id}-{index:03d}"
        used_ids.add(image_id)
        destination = output_dir / f"{image_id}.jpg"
        transform, original_size = normalize_image_file(source, destination, width, height)
        plan = plan_by_id.get(image_id, {})
        record = {
            "image_id": image_id,
            "source_path": str(source.resolve()),
            "working_path": str(destination.resolve()),
            "source_width": original_size[0],
            "source_height": original_size[1],
            "working_width": width,
            "working_height": height,
            "transform": transform.to_dict(),
            "purpose": "dataset" if plan else "visibility-only",
        }
        for key in ("session_id", "pot_target_id", "pot_target_deg", "s_left_on", "s_right_on", "split"):
            if key in plan:
                record[key] = plan[key]
        records.append(record)
    write_dataset_manifest(manifest_path, records)
    return {"images_imported": len(records), "manifest": str(manifest_path), "output_dir": str(output_dir)}


def _draw_annotation(image: np.ndarray, annotation: dict[str, Any] | None) -> np.ndarray:
    overlay = image.copy()
    if annotation is None:
        cv2.rectangle(overlay, (0, 0), (image.shape[1] - 1, 34), (0, 165, 255), -1)
        cv2.putText(overlay, "UNANNOTATED", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2, cv2.LINE_AA)
        return overlay
    status = annotation["status"]
    color = (40, 210, 40) if status == "valid" else (0, 165, 255)
    cv2.rectangle(overlay, (0, 0), (image.shape[1] - 1, 34), color, -1)
    label = f"{status.upper()}  L={annotation.get('s_left_on', '?')} R={annotation.get('s_right_on', '?')}"
    cv2.putText(overlay, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
    required_points = ("axis_tail", "axis_head", "pot_center", "pot_tip")
    if not all(key in annotation for key in required_points):
        return overlay
    axis_tail = tuple(round(value) for value in annotation["axis_tail"])
    axis_head = tuple(round(value) for value in annotation["axis_head"])
    center = tuple(round(value) for value in annotation["pot_center"])
    tip = tuple(round(value) for value in annotation["pot_tip"])
    cv2.arrowedLine(overlay, axis_tail, axis_head, (30, 230, 230), 3, tipLength=0.04)
    cv2.circle(overlay, center, 7, (0, 255, 0), 2)
    cv2.arrowedLine(overlay, center, tip, (0, 110, 255), 3, tipLength=0.2)
    return overlay


def _color_visibility(image: np.ndarray) -> dict[str, Any]:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    orange = cv2.inRange(hsv, np.array([5, 100, 80]), np.array([25, 255, 255]))
    blue = cv2.inRange(hsv, np.array([90, 70, 50]), np.array([135, 255, 255]))
    light_board = cv2.inRange(hsv, np.array([0, 0, 120]), np.array([180, 90, 255]))
    total = image.shape[0] * image.shape[1]
    counts = {
        "orange_pixels": int(cv2.countNonZero(orange)),
        "blue_pixels": int(cv2.countNonZero(blue)),
        "light_board_pixels": int(cv2.countNonZero(light_board)),
    }
    return {
        **counts,
        "orange_candidate_visible": counts["orange_pixels"] >= max(12, total // 50000),
        "blue_candidate_visible": counts["blue_pixels"] >= max(12, total // 50000),
        "board_candidate_visible": counts["light_board_pixels"] >= total // 20,
        "note": "Color thresholds are a diagnostic, not proof that pixels belong to the intended controls.",
    }


def preprocess_check(
    manifest_path: Path, output_dir: Path, annotations_path: Path | None = None
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {output_dir}")
    output_dir.mkdir(parents=True)
    payload = read_dataset_manifest(manifest_path)
    annotations = read_annotations(annotations_path) if annotations_path else {}
    checks: list[dict[str, Any]] = []
    for record in payload["images"]:
        image = cv2.imread(record["working_path"], cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"could not read working image: {record['working_path']}")
        if image.shape[1] != record["working_width"] or image.shape[0] != record["working_height"]:
            raise ValueError(f"working image size mismatch: {record['image_id']}")
        overlay = _draw_annotation(image, annotations.get(record["image_id"]))
        destination = output_dir / f"{record['image_id']}-overlay.jpg"
        if not cv2.imwrite(str(destination), overlay, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise OSError(f"could not write overlay: {destination}")
        checks.append(
            {
                "image_id": record["image_id"],
                "output": str(destination),
                "annotation_status": annotations.get(record["image_id"], {}).get("status", "unannotated"),
                "dimensions_rgb": [record["working_width"], record["working_height"], 3],
                "visibility_diagnostic": _color_visibility(image),
            }
        )
    report = {"schema_version": 1, "images": checks}
    (output_dir / "preprocess-report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
