from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from PIL import Image
import torch


def run_cli(tmp_path: Path, *arguments: str) -> dict:
    completed = subprocess.run(
        [sys.executable, "-m", "oscillator_cv", *arguments],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def test_every_cli_command(tmp_path: Path) -> None:
    plan = tmp_path / "capture-plan.csv"
    capture = run_cli(tmp_path, "capture-plan", "--output", str(plan))
    assert capture["images"] == 120
    assert capture["sessions"] == 3
    expanded = run_cli(tmp_path, "capture-plan", "--sessions", "6", "--output", str(tmp_path / "expanded.csv"))
    assert expanded["images"] == 240

    originals = tmp_path / "originals"
    originals.mkdir()
    source = originals / "fixture.jpg"
    Image.new("RGB", (80, 60), (230, 230, 215)).save(source)
    processed = tmp_path / "processed"
    manifest = tmp_path / "manifest.json"
    imported = run_cli(
        tmp_path,
        "import-images",
        str(originals),
        "--output-dir",
        str(processed),
        "--manifest",
        str(manifest),
        "--width",
        "32",
        "--height",
        "48",
    )
    assert imported["images_imported"] == 1

    script = tmp_path / "annotation-script.json"
    script.write_text(
        json.dumps(
            {
                "annotations": [
                    {
                        "image_id": "fixture",
                        "status": "valid",
                        "axis_tail": [16, 43],
                        "axis_head": [16, 5],
                        "pot_center": [14, 20],
                        "pot_tip": [22, 20],
                        "s_left_on": 0,
                        "s_right_on": 1,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    annotations = tmp_path / "annotations.json"
    annotated = run_cli(
        tmp_path, "annotate", str(manifest), "--output", str(annotations), "--script", str(script)
    )
    assert annotated["annotations_written"] == 1
    overlay_dir = tmp_path / "overlays"
    checked = run_cli(
        tmp_path,
        "preprocess-check",
        str(manifest),
        "--annotations",
        str(annotations),
        "--output-dir",
        str(overlay_dir),
    )
    assert checked["images"] == 1

    train_dir = tmp_path / "train"
    trained = run_cli(
        tmp_path,
        "train",
        "--synthetic",
        "--samples",
        "4",
        "--image-width",
        "32",
        "--image-height",
        "48",
        "--base-channels",
        "2",
        "--epochs",
        "1",
        "--batch-size",
        "2",
        "--seed",
        "23",
        "--output-dir",
        str(train_dir),
    )
    checkpoint = Path(trained["checkpoint"])
    assert checkpoint.exists()
    assert torch.load(checkpoint, weights_only=True)["training_seed"] == 23

    resumed_dir = tmp_path / "resume"
    resumed = run_cli(
        tmp_path,
        "train",
        "--synthetic",
        "--samples",
        "4",
        "--image-width",
        "32",
        "--image-height",
        "48",
        "--base-channels",
        "2",
        "--epochs",
        "1",
        "--batch-size",
        "2",
        "--resume",
        str(checkpoint),
        "--seed",
        "23",
        "--output-dir",
        str(resumed_dir),
    )
    resumed_checkpoint = Path(resumed["checkpoint"])
    assert "0002" in resumed_checkpoint.name

    evaluation = run_cli(
        tmp_path,
        "evaluate",
        str(resumed_checkpoint),
        "--synthetic",
        "--samples",
        "4",
        "--batch-size",
        "2",
        "--seed",
        "23",
    )
    assert evaluation["images"] == 4
    prediction_path = tmp_path / "prediction.json"
    prediction = run_cli(
        tmp_path,
        "predict",
        str(source),
        "--checkpoint",
        str(resumed_checkpoint),
        "--output",
        str(prediction_path),
    )
    assert prediction["schema_version"] == 1
    assert prediction["accepted"] is False
    assert set(prediction["prediction"]) == {"board_orientation", "switches", "potentiometer"}
    assert json.loads(prediction_path.read_text(encoding="utf-8")) == prediction
