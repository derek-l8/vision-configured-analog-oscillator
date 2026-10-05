from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from oscillator_cv.annotation_zoom import PotZoomView
from oscillator_cv.geometry import LetterboxTransform
from oscillator_cv.imaging import load_board_portrait_rgb
from oscillator_cv.manifest import read_annotations, read_dataset_manifest, write_annotations


POINT_KEYS = ("axis_tail", "axis_head", "pot_center", "pot_tip")


def import_scripted_annotations(manifest_path: Path, output_path: Path, script_path: Path) -> dict[str, Any]:
    manifest = read_dataset_manifest(manifest_path)
    valid_ids = {record["image_id"] for record in manifest["images"]}
    script = json.loads(script_path.read_text(encoding="utf-8"))
    incoming = script["annotations"] if isinstance(script, dict) else script
    annotations = read_annotations(output_path)
    for annotation in incoming:
        image_id = str(annotation["image_id"])
        if image_id not in valid_ids:
            raise ValueError(f"scripted annotation references unknown image_id: {image_id}")
        annotations[image_id] = annotation
    write_annotations(output_path, annotations)
    return {"annotations_written": len(incoming), "output": str(output_path)}


def annotate_interactive(manifest_path: Path, output_path: Path, review_pot: bool = False) -> dict[str, Any]:
    """OpenCV click-and-key interface.

    Click axis tail, axis head, then near the dial to open an original-resolution
    zoom. Click center and tip in the zoom. Review mode preserves axis/switches.
    Keys: 1/2 set left OFF/ON, 3/4 set right OFF/ON;
    v/o/a/r set valid/occluded/ambiguous/rejected; n saves; z resets; q quits.
    """
    manifest = read_dataset_manifest(manifest_path)
    annotations = read_annotations(output_path)
    completed = 0
    for record in manifest["images"]:
        previous = annotations.get(record["image_id"])
        if review_pot:
            if previous is None or previous["status"] != "valid":
                continue
            if previous.get("pot_annotation_method") == "original-resolution-zoom-v1":
                continue
        elif previous is not None:
            continue
        image = cv2.imread(record["working_path"], cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"could not load {record['working_path']}")
        original = load_board_portrait_rgb(Path(record["source_path"]))
        working_transform = LetterboxTransform(**record["transform"])
        points: list[tuple[float, float]] = []
        states: dict[str, Any] = {"s_left_on": None, "s_right_on": None, "status": "valid"}
        zoom: PotZoomView | None = None
        cursor: tuple[int, int] | None = None
        if review_pot and previous is not None:
            points = [tuple(previous[key]) for key in POINT_KEYS[:2]]
            states.update({key: previous[key] for key in states})
            zoom = PotZoomView.create(original, working_transform, tuple(previous["pot_center"]))

        def on_mouse(event: int, x: int, y: int, _flags: int, _param: object) -> None:
            nonlocal zoom, cursor
            cursor = (x, y)
            if event != cv2.EVENT_LBUTTONDOWN or len(points) >= 4 or y < 84:
                return
            if len(points) < 2:
                points.append((float(x), float(y)))
            elif zoom is None:
                zoom = PotZoomView.create(original, working_transform, (float(x), float(y)))
            else:
                point = zoom.display_to_working_point((float(x), float(y)))
                if point is not None:
                    points.append(point)
                    if len(points) == 4:
                        zoom = None

        window = "oscillator-cv annotate"
        cv2.namedWindow(window, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(window, 576, 768)
        cv2.setMouseCallback(window, on_mouse)
        while True:
            display = cv2.cvtColor(np.asarray(zoom.image), cv2.COLOR_RGB2BGR) if zoom else image.copy()
            for index, point in enumerate(points):
                if zoom and index < 2:
                    continue
                drawn = zoom.working_to_display_point(point) if zoom else point
                px, py = round(drawn[0]), round(drawn[1])
                cv2.drawMarker(display, (px, py), (0, 255, 255), cv2.MARKER_CROSS, 9, 1)
                cv2.putText(display, POINT_KEYS[index], (px + 8, py), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            if zoom and cursor:
                cv2.drawMarker(display, cursor, (255, 255, 255), cv2.MARKER_CROSS, 17, 1)
            if zoom:
                instruction = "Zoom: click dial CENTER" if len(points) == 2 else "Zoom: click orange arrow TIP"
            else:
                instruction = ("Click rail at BUZZER end", "Click SAME rail at POWER end",
                               "Click near dial to ZOOM", "Click orange arrow TIP", "n: save/next; e: redo pot; q: quit")[len(points)]
            lines = [record["image_id"], f"L={states['s_left_on']} R={states['s_right_on']} status={states['status']}",
                     instruction, "z: reset/recenter; q: quit"]
            cv2.rectangle(display, (0, 0), (display.shape[1] - 1, 83), (20, 20, 20), -1)
            for row, line in enumerate(lines):
                cv2.putText(display, line, (8, 17 + 20 * row), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (240, 240, 240), 1, cv2.LINE_AA)
            cv2.imshow(window, display)
            key = cv2.waitKey(30) & 0xFF
            if key == ord("q"):
                cv2.destroyAllWindows()
                return {"annotations_written": completed, "output": str(output_path), "quit_early": True}
            if key == ord("z"):
                points = points[:2] if review_pot else []
                zoom = None
            elif key == ord("e"):
                points = points[:2]
                zoom = None
            elif key == ord("1"):
                states["s_left_on"] = 0
            elif key == ord("2"):
                states["s_left_on"] = 1
            elif key == ord("3"):
                states["s_right_on"] = 0
            elif key == ord("4"):
                states["s_right_on"] = 1
            elif key in (ord("v"), ord("o"), ord("a"), ord("r")):
                states["status"] = {ord("v"): "valid", ord("o"): "occluded", ord("a"): "ambiguous", ord("r"): "rejected"}[key]
            elif key == ord("n"):
                is_valid = states["status"] == "valid"
                if is_valid and (len(points) != 4 or states["s_left_on"] is None or states["s_right_on"] is None):
                    continue
                annotation: dict[str, Any] = {**(previous or {}), "image_id": record["image_id"], **states}
                for point_key, point in zip(POINT_KEYS, points):
                    annotation[point_key] = list(point)
                if is_valid:
                    annotation["pot_annotation_method"] = "original-resolution-zoom-v1"
                annotations[record["image_id"]] = annotation
                write_annotations(output_path, annotations)
                completed += 1
                break
        cv2.destroyWindow(window)
    cv2.destroyAllWindows()
    return {"annotations_written": completed, "output": str(output_path), "quit_early": False}
