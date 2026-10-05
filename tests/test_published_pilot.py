import json
import shutil
from collections import Counter
from pathlib import Path

from PIL import Image

from oscillator_cv.dataset import AnnotatedImageDataset
from oscillator_cv.manifest import read_annotations, read_dataset_manifest


def test_published_pilot_is_complete_and_relocatable(tmp_path, monkeypatch) -> None:
    source = Path(__file__).resolve().parents[1] / "data" / "pilot-s01"
    package = tmp_path / "relocated-pilot"
    shutil.copytree(source, package)
    monkeypatch.chdir(tmp_path)
    manifest_path = package / "manifest.json"
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = read_dataset_manifest(manifest_path)["images"]
    labels = read_annotations(package / "annotations.json")
    assert len(records) == len(labels) == 40
    assert {record["image_id"] for record in records} == set(labels)
    assert {record["split"] for record in records} == {"train"}
    assert len({record["pot_target_id"] for record in records}) == 10
    assert Counter((r["s_left_on"], r["s_right_on"]) for r in records) == {
        (0, 0): 10, (1, 0): 10, (1, 1): 10, (0, 1): 10,
    }
    for raw_record, record in zip(raw["images"], records):
        for field in ("source_path", "working_path"):
            assert not Path(raw_record[field]).is_absolute()
            assert Path(record[field]).is_relative_to(package)
        assert record["source_path"] == record["working_path"]
        assert record["transform"]["scale"] == 1.0
        with Image.open(record["working_path"]) as image:
            assert image.size == (576, 768)
            assert image.mode == "RGB"
            assert not image.getexif()
            assert image.info == {}
        label = labels[record["image_id"]]
        assert label["status"] == "valid"
        assert (label["s_left_on"], label["s_right_on"]) == (
            record["s_left_on"], record["s_right_on"])
        for name in ("axis_tail", "axis_head", "pot_center", "pot_tip"):
            x, y = label[name]
            assert 0 <= x < 576 and 0 <= y < 768
    dataset = AnnotatedImageDataset(manifest_path, package / "annotations.json", split="train")
    assert len(dataset) == 40
    image, target, image_id = dataset[0]
    assert image.shape == (3, 768, 576)
    assert target["heatmaps"].shape == (2, 384, 288)
    assert image_id in labels
