"""Bounded real-image learning diagnostic; its metrics are training-only.

Run from the repository root with the local Python environment. No image, label,
existing run, or Git state is changed. The output directory must be new.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import torch
from PIL import Image, ImageDraw
from torch.utils.data import Subset

from oscillator_cv.dataset import AnnotatedImageDataset
from oscillator_cv.model import ModelConfig, MultiTaskCNN, model_from_checkpoint
from oscillator_cv.training import LOSS_VERSION, decode_batch, evaluate_model, load_checkpoint, multitask_loss, save_checkpoint


def diagnostic_passes(metrics: dict, samples: int) -> bool:
    return (metrics["exact_switch_configuration_accuracy"] == 1.0
            and metrics["orientation_mae_deg"] < 2.0
            and metrics["pot_center_mean_error_px"] < 3.0
            and metrics["pot_tip_mean_error_px"] < 3.0
            and metrics["pot_angle_mae_deg"] is not None
            and metrics["pot_angle_valid_images"] == samples
            and metrics["pot_angle_mae_deg"] < 15.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--check-every", type=int, default=25)
    parser.add_argument("--pot-ids", nargs="+", default=["p00", "p09"])
    parser.add_argument("--normalization", choices=("group", "batch"), default="group")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--switch-pool-grid", type=int, default=4)
    parser.add_argument("--evaluate-checkpoint", type=Path, help="render/evaluate this diagnostic checkpoint without training")
    args = parser.parse_args()
    if min(args.steps, args.check_every, args.batch_size, args.threads) <= 0:
        parser.error("steps, check-every, batch-size and threads must be positive")
    torch.set_num_threads(args.threads)
    dataset = AnnotatedImageDataset(args.manifest.resolve(), args.annotations.resolve(), split="train")
    indices = [i for i, (record, _) in enumerate(dataset.records) if record["image_id"].split("_")[1] in args.pot_ids]
    if not indices:
        parser.error("no training images match pot-ids")
    if args.output_dir.exists():
        parser.error("output directory already exists; refusing to overwrite")
    args.output_dir.mkdir(parents=True)
    subset = Subset(dataset, indices)
    # Cache only the selected eight images; don't repeatedly decode JPEGs.
    samples = [subset[i] for i in range(len(subset))]
    images = torch.stack([sample[0] for sample in samples])
    targets = {key: torch.stack([sample[1][key] for sample in samples]) for key in samples[0][1]}
    torch.manual_seed(11)
    model = MultiTaskCNN(ModelConfig(input_width=images.shape[3], input_height=images.shape[2],
                                    normalization=args.normalization, switch_pool_grid=args.switch_pool_grid))
    step = 0
    progress_unit = "optimization_steps"
    if args.evaluate_checkpoint:
        existing = load_checkpoint(args.evaluate_checkpoint, torch.device("cpu"))
        model = model_from_checkpoint(existing, torch.device("cpu"))
        step = int(existing["epoch"])
        progress_unit = ("optimization_steps" if existing.get("training_source") == "annotated-images-overfit-diagnostic" else "epochs")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    history = list(existing.get("history", [])) if args.evaluate_checkpoint else []
    started = time.monotonic()
    passed = False
    metrics = evaluate_model(model, subset, batch_size=args.batch_size)
    print(json.dumps({"completed": step, "progress_unit": progress_unit, "metrics": metrics}), flush=True)
    for step in range(1, (0 if args.evaluate_checkpoint else args.steps) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        chosen = torch.randperm(len(samples))[:args.batch_size]
        batch_targets = {key: value[chosen] for key, value in targets.items()}
        loss, components = multitask_loss(model(images[chosen]), batch_targets)
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite training loss")
        loss.backward()
        optimizer.step()
        history.append({"step": step, **components})
        if step % args.check_every == 0 or step == args.steps:
            metrics = evaluate_model(model, subset, batch_size=args.batch_size)
            passed = diagnostic_passes(metrics, len(samples))
            print(json.dumps({"step": step, "loss": components, "metrics": metrics,
                              "diagnostic_gate_passed": passed, "elapsed_seconds": time.monotonic() - started}), flush=True)
            save_checkpoint(args.output_dir / f"checkpoint-step-{step:04d}.pt", model, optimizer, step, history,
                            "annotated-images-overfit-diagnostic", seed=11)
            if passed:
                break
    passed = diagnostic_passes(metrics, len(samples))
    checkpoint = args.evaluate_checkpoint or args.output_dir / "checkpoint-final.pt"
    # The diagnostic stores optimization steps in the checkpoint's epoch field.
    # Do not resume it through the ordinary epoch-based training command.
    if not args.evaluate_checkpoint:
        save_checkpoint(checkpoint, model, optimizer, step, history, "annotated-images-overfit-diagnostic", seed=11)
    model.eval()
    predictions = []
    with torch.no_grad():
        for offset in range(0, len(samples), args.batch_size):
            predictions.extend(decode_batch(model(images[offset:offset+args.batch_size])))
    per_image = []
    tiles = []
    closeups = []
    for i, prediction in enumerate(predictions):
        record, annotation = dataset.records[indices[i]]
        per_image.append({"image_id": record["image_id"], "prediction": prediction, "annotation": annotation})
        with Image.open(record["working_path"]) as source:
            tile = source.convert("RGB")
        draw = ImageDraw.Draw(tile)
        for key, predicted in (("pot_center", prediction["potentiometer"]["center_xy"]),
                               ("pot_tip", prediction["potentiometer"]["tip_xy"])):
            x, y = annotation[key]
            draw.ellipse((x-4, y-4, x+4, y+4), outline="lime", width=2)
            x, y = predicted
            draw.line((x-5, y, x+5, y), fill="magenta", width=2)
            draw.line((x, y-5, x, y+5), fill="magenta", width=2)
        draw.rectangle((0, 0, tile.width, 42), fill="black")
        draw.text((8, 4), record["image_id"], fill="white")
        draw.text((8, 22), "labels: green circles; predictions: magenta crosses", fill="white")
        tile.save(args.output_dir / f"{record['image_id']}-prediction.jpg", quality=95)
        cx, cy = annotation["pot_center"]
        closeup = Image.new("RGB", (192, 192), "black")
        crop = tile.crop((round(cx-32), round(cy-32), round(cx+32), round(cy+32))).resize((192, 160))
        closeup.paste(crop, (0, 32))
        closeup_draw = ImageDraw.Draw(closeup)
        closeup_draw.text((4, 2), record["image_id"], fill="white")
        expected = (annotation["s_left_on"], annotation["s_right_on"])
        actual = (prediction["switches"]["left"]["state"], prediction["switches"]["right"]["state"])
        closeup_draw.text((4, 16), f"switch truth={expected} pred={actual}", fill="white" if expected == actual else "red")
        closeups.append(closeup)
        tiles.append(tile.resize((288, 384)))
    sheet = Image.new("RGB", (4*288, ((len(tiles)+3)//4)*384))
    for i, tile in enumerate(tiles):
        sheet.paste(tile, ((i%4)*288, (i//4)*384))
    sheet.save(args.output_dir / "predictions-contact-sheet.jpg", quality=95)
    detail_sheet = Image.new("RGB", (5*192, ((len(closeups)+4)//5)*192))
    for i, closeup in enumerate(closeups):
        detail_sheet.paste(closeup, ((i%5)*192, (i//5)*192))
    detail_sheet.save(args.output_dir / "pot-predictions-contact-sheet.jpg", quality=95)
    report = {"loss_version": LOSS_VERSION, "training_progress": {"unit": progress_unit, "completed": step},
              "training_images": len(samples),
              "seed": 11,
              "normalization": model.config.normalization, "switch_pool_grid": model.config.switch_pool_grid,
              "batch_size": args.batch_size, "evaluation_only": bool(args.evaluate_checkpoint),
              "image_ids": [sample[2] for sample in samples], "evaluation_scope": "same images used for training",
              "diagnostic_gate_passed": passed, "accepted_for_inference": False,
              "elapsed_seconds": time.monotonic()-started, "metrics": metrics,
              "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
              "annotations_sha256": hashlib.sha256(args.annotations.read_bytes()).hexdigest(),
              "checkpoint": str(checkpoint.resolve()), "predictions": per_image}
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    (args.output_dir / "history.json").write_text(json.dumps(history, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"report": str((args.output_dir / "report.json").resolve()), "diagnostic_gate_passed": passed}), flush=True)


if __name__ == "__main__":
    main()
