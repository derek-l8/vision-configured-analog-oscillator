from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

from oscillator_cv.annotation import annotate_interactive, import_scripted_annotations
from oscillator_cv.dataset import AnnotatedImageDataset, SyntheticOscillatorDataset
from oscillator_cv.imaging import letterbox_image, load_board_portrait_rgb, pil_to_tensor_array
from oscillator_cv.manifest import capture_rows, validate_capture_plan, write_capture_plan
from oscillator_cv.model import ModelConfig, model_from_checkpoint
from oscillator_cv.preprocess import import_images, preprocess_check
from oscillator_cv.training import decode_batch, evaluate_model, load_checkpoint, train_model


def _path(value: str) -> Path:
    return Path(value).resolve()


def _print_json(payload: Any) -> None:
    print(json.dumps(payload, indent=2))


def _dataset_from_args(args: argparse.Namespace, for_evaluation: bool = False):
    if args.synthetic:
        return SyntheticOscillatorDataset(
            samples=args.samples,
            width=args.image_width,
            height=args.image_height,
            seed=args.seed,
        )
    if not args.manifest or not args.annotations:
        raise ValueError("real-image operation requires --manifest and --annotations")
    split = args.split if getattr(args, "split", None) else ("test" if for_evaluation else "train")
    return AnnotatedImageDataset(_path(args.manifest), _path(args.annotations), split=split)


def command_capture_plan(args: argparse.Namespace) -> None:
    rows = capture_rows(args.sessions)
    validate_capture_plan(rows)
    output = _path(args.output)
    write_capture_plan(output, args.sessions)
    _print_json(
        {
            "output": str(output),
            "images": len(rows),
            "sessions": args.sessions,
            "potentiometer_targets_deg": list(range(45, 316, 30)),
            "switch_combinations": [[0, 0], [1, 0], [1, 1], [0, 1]],
        }
    )


def command_import_images(args: argparse.Namespace) -> None:
    result = import_images(
        _path(args.source_dir),
        _path(args.output_dir),
        _path(args.manifest),
        _path(args.capture_plan) if args.capture_plan else None,
        args.width,
        args.height,
    )
    _print_json(result)


def command_annotate(args: argparse.Namespace) -> None:
    if args.script:
        result = import_scripted_annotations(_path(args.manifest), _path(args.output), _path(args.script))
    else:
        result = annotate_interactive(_path(args.manifest), _path(args.output), review_pot=args.review_pot)
    _print_json(result)


def command_preprocess_check(args: argparse.Namespace) -> None:
    result = preprocess_check(
        _path(args.manifest),
        _path(args.output_dir),
        _path(args.annotations) if args.annotations else None,
    )
    _print_json({"images": len(result["images"]), "output_dir": str(_path(args.output_dir))})


def command_train(args: argparse.Namespace) -> None:
    dataset = _dataset_from_args(args)
    sample_image, _, _ = dataset[0]
    config = ModelConfig(
        base_channels=args.base_channels,
        normalization=args.normalization,
        switch_pool_grid=args.switch_pool_grid,
        input_width=int(sample_image.shape[2]),
        input_height=int(sample_image.shape[1]),
    )
    checkpoint = train_model(
        dataset=dataset,
        output_dir=_path(args.output_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        model_config=config,
        resume=_path(args.resume) if args.resume else None,
        device_name=args.device,
        training_source="synthetic-smoke-fixtures" if args.synthetic else "annotated-images",
        seed=args.seed,
    )
    _print_json({"checkpoint": str(checkpoint), "training_source": "synthetic" if args.synthetic else "annotated-images"})


def command_evaluate(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    checkpoint = load_checkpoint(_path(args.checkpoint), device)
    config = checkpoint["model_config"]
    if args.synthetic:
        args.image_width = config["input_width"]
        args.image_height = config["input_height"]
    dataset = _dataset_from_args(args, for_evaluation=True)
    model = model_from_checkpoint(checkpoint, device)
    metrics = evaluate_model(model, dataset, batch_size=args.batch_size)
    metrics["checkpoint"] = str(_path(args.checkpoint))
    metrics["checkpoint_training_source"] = checkpoint.get("training_source", "unknown")
    output = _path(args.output) if args.output else None
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    _print_json(metrics)


def command_predict(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    checkpoint_path = _path(args.checkpoint)
    checkpoint = load_checkpoint(checkpoint_path, device)
    model = model_from_checkpoint(checkpoint, device)
    model.eval()
    config = model.config
    source = _path(args.image)
    original = load_board_portrait_rgb(source)
    working, transform = letterbox_image(original, config.input_width, config.input_height)
    tensor = torch.from_numpy(pil_to_tensor_array(working).copy()).unsqueeze(0).to(device)
    with torch.no_grad():
        prediction = decode_batch(model(tensor))[0]
    result = {
        "schema_version": 1,
        "image": str(source),
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoint_training_source": checkpoint.get("training_source", "unknown"),
        "working_size": [config.input_width, config.input_height],
        "preprocess_transform": transform.to_dict(),
        "prediction": prediction,
        "accepted": False,
        "acceptance_reason": "No confidence thresholds or real held-out validation are established.",
        "potentiometer_calibration_applied": False,
    }
    if args.output:
        output = _path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    _print_json(result)


def _add_dataset_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--manifest")
    parser.add_argument("--annotations")
    parser.add_argument("--split")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--image-width", type=int, default=96)
    parser.add_argument("--image-height", type=int, default=128)
    parser.add_argument("--seed", type=int, default=11, help="training RNG and synthetic fixture seed (default: 11)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="oscillator-cv")
    subparsers = parser.add_subparsers(dest="command", required=True)

    capture = subparsers.add_parser("capture-plan", help="generate a three-session pilot or six-session expanded capture plan")
    capture.add_argument("--output", default="data/manifests/pilot-capture-plan.csv")
    capture.add_argument("--sessions", type=int, choices=(3, 6), default=3)
    capture.set_defaults(function=command_capture_plan)

    image_import = subparsers.add_parser("import-images", help="normalize JPEG orientation and build a dataset manifest")
    image_import.add_argument("source_dir")
    image_import.add_argument("--output-dir", required=True)
    image_import.add_argument("--manifest", required=True)
    image_import.add_argument("--capture-plan")
    image_import.add_argument("--width", type=int, default=576)
    image_import.add_argument("--height", type=int, default=768)
    image_import.set_defaults(function=command_import_images)

    annotate = subparsers.add_parser("annotate", help="annotate board axis, potentiometer points, and switch states")
    annotate.add_argument("manifest")
    annotate.add_argument("--output", required=True)
    annotate.add_argument("--script", help="noninteractive JSON annotations for reproducible tests or reviewed imports")
    annotate.add_argument("--review-pot", action="store_true", help="refine saved center/tip labels in original-resolution zoom, retaining axis/switches")
    annotate.set_defaults(function=command_annotate)

    check = subparsers.add_parser("preprocess-check", help="render normalized images and annotation overlays")
    check.add_argument("manifest")
    check.add_argument("--annotations")
    check.add_argument("--output-dir", required=True)
    check.set_defaults(function=command_preprocess_check)

    train = subparsers.add_parser("train", help="train or resume the custom multitask CNN")
    _add_dataset_arguments(train)
    train.add_argument("--output-dir", required=True)
    train.add_argument("--resume")
    train.add_argument("--epochs", type=int, default=10)
    train.add_argument("--batch-size", type=int, default=4)
    train.add_argument("--learning-rate", type=float, default=1e-3)
    train.add_argument("--base-channels", type=int, default=16)
    train.add_argument("--normalization", choices=("group", "batch"), default="group")
    train.add_argument("--switch-pool-grid", type=int, default=4)
    train.add_argument("--device", default="cpu")
    train.set_defaults(function=command_train)

    evaluate = subparsers.add_parser("evaluate", help="evaluate orientation, switches, and potentiometer keypoints/angle")
    evaluate.add_argument("checkpoint")
    _add_dataset_arguments(evaluate)
    evaluate.add_argument("--batch-size", type=int, default=4)
    evaluate.add_argument("--device", default="cpu")
    evaluate.add_argument("--output")
    evaluate.set_defaults(function=command_evaluate)

    predict = subparsers.add_parser("predict", help="emit structured JSON for one image")
    predict.add_argument("image")
    predict.add_argument("--checkpoint", required=True)
    predict.add_argument("--device", default="cpu")
    predict.add_argument("--output")
    predict.set_defaults(function=command_predict)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.function(args)
    except (ValueError, OSError, FileExistsError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
