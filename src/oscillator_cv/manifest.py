from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
POT_TARGETS_DEG = tuple(range(45, 316, 30))
SWITCH_ORDER = ((0, 0), (1, 0), (1, 1), (0, 1))
VALID_STATUSES = {"valid", "occluded", "ambiguous", "rejected"}


def capture_rows(session_count: int = 3) -> list[dict[str, Any]]:
    if session_count not in (3, 6):
        raise ValueError("capture plan must contain 3 or 6 sessions")
    rows: list[dict[str, Any]] = []
    sequence = 1
    rotation_bins = (-25, -15, -5, 5, 15, 25)
    for session_index in range(1, session_count + 1):
        session_id = f"s{session_index:02d}"
        for pot_index, target_deg in enumerate(POT_TARGETS_DEG):
            for switch_index, (left, right) in enumerate(SWITCH_ORDER):
                image_id = f"{session_id}_p{pot_index:02d}_l{left}_r{right}_{sequence:03d}"
                rotation = rotation_bins[(session_index + pot_index + switch_index) % len(rotation_bins)]
                rows.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "image_id": image_id,
                        "relative_path": f"{image_id}.jpg",
                        "session_id": session_id,
                        "build_revision": "fixed-build-v1",
                        "pot_target_id": f"p{pot_index:02d}",
                        "pot_target_deg": target_deg,
                        "s_left_on": left,
                        "s_right_on": right,
                        "capture_rotation_intent_deg": rotation,
                        "camera_source": "iphone-rear-1x",
                        "capture_notes": "",
                        # Keep the pilot's held-out sessions when adding training data.
                        "split": "validation" if session_index == 2 else ("test" if session_index == 3 else "train"),
                    }
                )
                sequence += 1
    return rows


def write_capture_plan(path: Path, session_count: int = 3) -> None:
    rows = capture_rows(session_count)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".json":
        path.write_text(json.dumps({"schema_version": SCHEMA_VERSION, "images": rows}, indent=2) + "\n", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_capture_plan(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload["images"]
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in ("schema_version", "pot_target_deg", "s_left_on", "s_right_on", "capture_rotation_intent_deg"):
            row[key] = int(row[key])
    return rows


def validate_capture_plan(rows: list[dict[str, Any]], require_complete: bool = True) -> None:
    required = {
        "schema_version", "image_id", "relative_path", "session_id", "pot_target_id",
        "pot_target_deg", "s_left_on", "s_right_on", "split",
    }
    seen: set[str] = set()
    session_splits: dict[str, str] = {}
    for index, row in enumerate(rows):
        missing = required - row.keys()
        if missing:
            raise ValueError(f"row {index} missing fields: {sorted(missing)}")
        image_id = str(row["image_id"])
        if image_id in seen:
            raise ValueError(f"duplicate image_id: {image_id}")
        seen.add(image_id)
        if int(row["schema_version"]) != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version in {image_id}")
        if int(row["s_left_on"]) not in (0, 1) or int(row["s_right_on"]) not in (0, 1):
            raise ValueError(f"switch states must be binary in {image_id}")
        session, split = str(row["session_id"]), str(row["split"])
        if split not in {"train", "validation", "test"}:
            raise ValueError(f"invalid split in {image_id}: {split}")
        if session in session_splits and session_splits[session] != split:
            raise ValueError(f"session {session} must belong to one split")
        session_splits[session] = split
    if require_complete:
        if len(rows) not in (120, 240):
            raise ValueError(f"capture plan must contain 120 or 240 rows, found {len(rows)}")
        sessions = {str(row["session_id"]) for row in rows}
        session_count = len(rows) // 40
        if sessions != {f"s{i:02d}" for i in range(1, session_count + 1)}:
            raise ValueError(f"capture plan must contain sessions s01-s{session_count:02d}, found {sorted(sessions)}")
        for session in sessions:
            session_rows = [row for row in rows if row["session_id"] == session]
            observed = {
                (int(row["pot_target_deg"]), int(row["s_left_on"]), int(row["s_right_on"]))
                for row in session_rows
            }
            expected = {(angle, left, right) for angle in POT_TARGETS_DEG for left, right in SWITCH_ORDER}
            if observed != expected:
                raise ValueError(f"session {session} does not contain the complete 10x4 grid")


def read_dataset_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    validate_dataset_manifest(payload)
    # Published datasets use paths relative to the manifest, not the caller's cwd.
    base = path.resolve().parent
    for record in payload["images"]:
        for field in ("source_path", "working_path"):
            image_path = Path(record[field])
            if not image_path.is_absolute():
                record[field] = str((base / image_path).resolve())
    return payload


def write_dataset_manifest(path: Path, images: list[dict[str, Any]]) -> None:
    payload = {"schema_version": SCHEMA_VERSION, "images": images}
    validate_dataset_manifest(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def validate_dataset_manifest(payload: dict[str, Any]) -> None:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("dataset manifest has unsupported schema_version")
    images = payload.get("images")
    if not isinstance(images, list):
        raise ValueError("dataset manifest images must be a list")
    seen: set[str] = set()
    for index, record in enumerate(images):
        required = {"image_id", "source_path", "working_path", "transform", "working_width", "working_height"}
        missing = required - record.keys()
        if missing:
            raise ValueError(f"image record {index} missing fields: {sorted(missing)}")
        image_id = str(record["image_id"])
        if image_id in seen:
            raise ValueError(f"duplicate dataset image_id: {image_id}")
        seen.add(image_id)
        if int(record["working_width"]) <= 0 or int(record["working_height"]) <= 0:
            raise ValueError(f"invalid working size in {image_id}")


def read_annotations(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA_VERSION or not isinstance(payload.get("annotations"), list):
        raise ValueError("annotations file has invalid schema")
    result: dict[str, dict[str, Any]] = {}
    for annotation in payload["annotations"]:
        status = annotation.get("status")
        if status not in VALID_STATUSES:
            raise ValueError(f"invalid annotation status: {status}")
        result[str(annotation["image_id"])] = annotation
    return result


def write_annotations(path: Path, annotations: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema_version": SCHEMA_VERSION, "annotations": list(annotations.values())}
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
