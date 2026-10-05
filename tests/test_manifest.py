import copy
import json
from pathlib import Path

import pytest

from oscillator_cv.manifest import POT_TARGETS_DEG, capture_rows, read_dataset_manifest, validate_capture_plan, validate_dataset_manifest


@pytest.mark.parametrize("session_count", [3, 6])
def test_capture_plan_is_complete_and_grouped(session_count: int) -> None:
    rows = capture_rows(session_count)
    validate_capture_plan(rows)
    assert len(rows) == session_count * 40
    assert tuple(sorted({row["pot_target_deg"] for row in rows})) == POT_TARGETS_DEG
    assert {row["session_id"] for row in rows} == {f"s{i:02d}" for i in range(1, session_count + 1)}
    assert {row["split"] for row in rows if row["session_id"] == "s02"} == {"validation"}
    assert {row["split"] for row in rows if row["session_id"] == "s03"} == {"test"}
    assert sum(row["split"] == "train" for row in rows) == (session_count - 2) * 40
    assert sum(row["split"] == "test" for row in rows) == 40


def test_expansion_preserves_pilot_ids_and_splits() -> None:
    assert capture_rows(6)[:120] == capture_rows(3)
    with pytest.raises(ValueError, match="3 or 6"):
        capture_rows(4)


@pytest.mark.parametrize("split", ["test", "unknown"])
def test_capture_plan_rejects_mixed_or_invalid_session_splits(split: str) -> None:
    rows = capture_rows()
    rows[1]["split"] = split
    with pytest.raises(ValueError, match="one split|invalid split"):
        validate_capture_plan(rows)


def test_import_uses_pilot_session_splits(tmp_path) -> None:
    from PIL import Image
    from oscillator_cv.manifest import read_dataset_manifest, write_capture_plan
    from oscillator_cv.preprocess import import_images

    plan = tmp_path / "plan.csv"
    write_capture_plan(plan)
    originals = tmp_path / "originals"
    originals.mkdir()
    rows = capture_rows()
    for index in (0, 40, 80):
        Image.new("RGB", (32, 48)).save(originals / rows[index]["relative_path"])
    manifest = tmp_path / "manifest.json"
    import_images(originals, tmp_path / "processed", manifest, plan, width=32, height=48)
    records = read_dataset_manifest(manifest)["images"]
    assert {row["session_id"]: row["split"] for row in records} == {
        "s01": "train", "s02": "validation", "s03": "test",
    }


def test_capture_plan_rejects_duplicates() -> None:
    rows = capture_rows()
    bad_rows = copy.deepcopy(rows)
    bad_rows[1]["image_id"] = bad_rows[0]["image_id"]
    with pytest.raises(ValueError, match="duplicate image_id"):
        validate_capture_plan(bad_rows)


def test_dataset_manifest_validation() -> None:
    payload = {
        "schema_version": 1,
        "images": [
            {
                "image_id": "a",
                "source_path": "source.jpg",
                "working_path": "working.jpg",
                "working_width": 576,
                "working_height": 768,
                "transform": {},
            }
        ],
    }
    validate_dataset_manifest(payload)
    payload["images"].append(dict(payload["images"][0]))
    with pytest.raises(ValueError, match="duplicate"):
        validate_dataset_manifest(payload)


@pytest.mark.parametrize("absolute", [False, True])
def test_dataset_paths_resolve_from_manifest_not_cwd(tmp_path, monkeypatch, absolute) -> None:
    package = tmp_path / "dataset"
    package.mkdir()
    image = package / "images" / "a.jpg"
    stored_path = str(image) if absolute else "images/a.jpg"
    record = dict(image_id="a", source_path=stored_path, working_path=stored_path,
                  working_width=576, working_height=768, transform={})
    manifest = package / "manifest.json"
    serialized = json.dumps(dict(schema_version=1, images=[record]))
    manifest.write_text(serialized, encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    loaded = read_dataset_manifest(Path("dataset/manifest.json"))["images"][0]
    assert Path(loaded["source_path"]) == image.resolve()
    assert Path(loaded["working_path"]) == image.resolve()
    assert manifest.read_text(encoding="utf-8") == serialized
