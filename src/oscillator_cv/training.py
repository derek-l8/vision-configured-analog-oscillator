from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from oscillator_cv.geometry import circular_error_degrees, degrees_from_sin_cos, potentiometer_angle_degrees
from oscillator_cv.model import ModelConfig, MultiTaskCNN, model_from_checkpoint


LOSS_VERSION = "spatial-kl-v1"


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    """Load tensor/dictionary checkpoints without unrestricted pickle execution."""
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except pickle.UnpicklingError as error:
        raise ValueError("checkpoint contains unsupported serialized objects; only tensor/dictionary checkpoints are accepted") from error
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint must be a dictionary")
    return checkpoint


def spatial_heatmap_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """KL divergence between each normalized Gaussian and a spatial softmax.

    Each keypoint contributes equally regardless of image area. A uniformly
    suppressed map cannot improve this loss: the competition is between locations,
    not between the tiny foreground and the much larger zero-valued background.
    """
    if logits.shape != targets.shape:
        raise ValueError("heatmap prediction and target shapes must match")
    mass = targets.flatten(2).sum(-1, keepdim=True)
    if torch.any(mass <= 0):
        raise ValueError("every target heatmap must contain a keypoint")
    distribution = targets.flatten(2) / mass
    log_probabilities = F.log_softmax(logits.flatten(2), dim=-1)
    return F.kl_div(log_probabilities, distribution, reduction="none").sum(-1).mean()


def multitask_loss(
    outputs: dict[str, torch.Tensor], targets: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, dict[str, float]]:
    orientation_loss = F.mse_loss(outputs["orientation_sin_cos"], targets["orientation_sin_cos"])
    switch_loss = F.binary_cross_entropy_with_logits(outputs["switch_logits"], targets["switch_states"])
    heatmap_loss = spatial_heatmap_loss(outputs["heatmap_logits"], targets["heatmaps"])
    total = orientation_loss + switch_loss + heatmap_loss
    return total, {
        "total": float(total.detach()),
        "orientation": float(orientation_loss.detach()),
        "switch": float(switch_loss.detach()),
        "heatmap": float(heatmap_loss.detach()),
    }


def save_checkpoint(
    path: Path,
    model: MultiTaskCNN,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    history: list[dict[str, float]],
    training_source: str,
    seed: int | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "epoch": epoch,
            "model_config": model.config.to_dict(),
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "history": history,
            "training_source": training_source,
            "loss_version": LOSS_VERSION,
            "training_seed": seed,
        },
        path,
    )


def load_training_checkpoint(
    path: Path, device: torch.device, learning_rate: float
) -> tuple[MultiTaskCNN, torch.optim.Optimizer, int, list[dict[str, float]]]:
    checkpoint = load_checkpoint(path, device)
    if checkpoint.get("loss_version") != LOSS_VERSION:
        raise ValueError("checkpoint used a different heatmap loss; start a new run instead of resuming it")
    if checkpoint.get("training_source") == "annotated-images-overfit-diagnostic":
        raise ValueError("overfit checkpoint counts optimization steps, not epochs; start a separate ordinary training run")
    model = model_from_checkpoint(checkpoint, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    for group in optimizer.param_groups:
        group["lr"] = learning_rate
    return model, optimizer, int(checkpoint["epoch"]), list(checkpoint.get("history", []))


def train_model(
    dataset: Dataset,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    model_config: ModelConfig,
    resume: Path | None = None,
    device_name: str = "cpu",
    training_source: str = "annotated-images",
    seed: int = 11,
) -> Path:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing run directory: {output_dir}")
    output_dir.mkdir(parents=True)
    device = torch.device(device_name)
    torch.manual_seed(seed)
    if resume:
        model, optimizer, start_epoch, history = load_training_checkpoint(resume, device, learning_rate)
        if model.config != model_config:
            raise ValueError("resume checkpoint model configuration does not match the requested configuration")
    else:
        model = MultiTaskCNN(model_config).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        start_epoch = 0
        history = []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    model.train()
    last_checkpoint: Path | None = None
    for epoch in range(start_epoch + 1, start_epoch + epochs + 1):
        sums = {key: 0.0 for key in ("total", "orientation", "switch", "heatmap")}
        batches = 0
        for images, targets, _ in loader:
            images = images.to(device)
            tensor_targets = {key: value.to(device) for key, value in targets.items()}
            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss, components = multitask_loss(outputs, tensor_targets)
            loss.backward()
            optimizer.step()
            for key, value in components.items():
                sums[key] += value
            batches += 1
        epoch_record = {"epoch": float(epoch), **{key: value / batches for key, value in sums.items()}}
        history.append(epoch_record)
        print(json.dumps(epoch_record), file=sys.stderr, flush=True)
        last_checkpoint = output_dir / f"checkpoint-epoch-{epoch:04d}.pt"
        save_checkpoint(last_checkpoint, model, optimizer, epoch, history, training_source, seed=seed)
    assert last_checkpoint is not None
    (output_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    return last_checkpoint


def heatmap_argmax_points(logits: torch.Tensor) -> torch.Tensor:
    """Locate peaks and refine each with a local quadratic fit in log space.

    Targets are Gaussians on a half-resolution grid. Their log is quadratic,
    allowing subpixel coordinates instead of rounding every point to two pixels.
    Flat, edge, or non-concave peaks fall back to the grid location.
    """
    batch, channels, height, width = logits.shape
    flat_indices = logits.reshape(batch, channels, -1).argmax(dim=-1)
    y = torch.div(flat_indices, width, rounding_mode="floor")
    x = flat_indices % width
    refined = torch.stack((x.float(), y.float()), dim=-1)
    for sample in range(batch):
        for channel in range(channels):
            px, py = int(x[sample, channel]), int(y[sample, channel])
            for axis, coordinate, extent in ((0, px, width), (1, py, height)):
                if coordinate == 0 or coordinate == extent - 1:
                    continue
                peak = logits[sample, channel, py, px]
                lower = logits[sample, channel, py, px-1] if axis == 0 else logits[sample, channel, py-1, px]
                upper = logits[sample, channel, py, px+1] if axis == 0 else logits[sample, channel, py+1, px]
                curvature = lower - 2*peak + upper
                if torch.isfinite(curvature) and curvature < -1e-6:
                    offset = (0.5 * (lower-upper) / curvature).clamp(-0.5, 0.5)
                    refined[sample, channel, axis] += offset
    return refined * 2.0


def decode_batch(outputs: dict[str, torch.Tensor]) -> list[dict[str, Any]]:
    orientations = outputs["orientation_sin_cos"].detach().cpu()
    switch_probs = torch.sigmoid(outputs["switch_logits"]).detach().cpu()
    heatmap_logits = outputs["heatmap_logits"].detach().cpu()
    heatmap_probs = heatmap_logits.flatten(2).softmax(-1).reshape_as(heatmap_logits)
    points = heatmap_argmax_points(heatmap_logits)
    results: list[dict[str, Any]] = []
    for index in range(orientations.shape[0]):
        sin_value, cos_value = orientations[index].tolist()
        rotation = degrees_from_sin_cos(sin_value, cos_value)
        center = tuple(points[index, 0].tolist())
        tip = tuple(points[index, 1].tolist())
        pot_angle = potentiometer_angle_degrees(center, tip, rotation) if center != tip else None
        results.append(
            {
                "board_orientation": {
                    "sin": sin_value,
                    "cos": cos_value,
                    "degrees": rotation,
                },
                "switches": {
                    "left": {"probability_on": float(switch_probs[index, 0]), "state": int(switch_probs[index, 0] >= 0.5)},
                    "right": {"probability_on": float(switch_probs[index, 1]), "state": int(switch_probs[index, 1] >= 0.5)},
                },
                "potentiometer": {
                    "center_xy": list(center),
                    "tip_xy": list(tip),
                    "board_relative_angle_deg": pot_angle,
                    "center_heatmap_peak": float(heatmap_probs[index, 0].max()),
                    "tip_heatmap_peak": float(heatmap_probs[index, 1].max()),
                },
            }
        )
    return results


def evaluate_model(model: nn.Module, dataset: Dataset, batch_size: int = 4) -> dict[str, Any]:
    device = next(model.parameters()).device
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    orientation_errors: list[float] = []
    pot_errors: list[float] = []
    point_errors = {"center": [], "tip": []}
    switch_correct = [0, 0]
    switch_counts = [0, 0]
    switch_tp = [0, 0]
    switch_fp = [0, 0]
    switch_fn = [0, 0]
    exact_correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for images, targets, _ in loader:
            outputs = model(images.to(device))
            decoded = decode_batch(outputs)
            for i, prediction in enumerate(decoded):
                target_rotation = float(targets["board_rotation_deg"][i])
                orientation_errors.append(circular_error_degrees(prediction["board_orientation"]["degrees"], target_rotation))
                predicted_states = [prediction["switches"]["left"]["state"], prediction["switches"]["right"]["state"]]
                target_states = [int(targets["switch_states"][i, 0]), int(targets["switch_states"][i, 1])]
                exact_correct += int(predicted_states == target_states)
                for switch_index in range(2):
                    switch_correct[switch_index] += int(predicted_states[switch_index] == target_states[switch_index])
                    switch_counts[switch_index] += 1
                    switch_tp[switch_index] += int(predicted_states[switch_index] == 1 and target_states[switch_index] == 1)
                    switch_fp[switch_index] += int(predicted_states[switch_index] == 1 and target_states[switch_index] == 0)
                    switch_fn[switch_index] += int(predicted_states[switch_index] == 0 and target_states[switch_index] == 1)
                for point_index, name in enumerate(("center", "tip")):
                    predicted_point = torch.tensor(prediction["potentiometer"][f"{name}_xy"])
                    target_point = targets["points"][i, point_index]
                    point_errors[name].append(float(torch.linalg.vector_norm(predicted_point - target_point)))
                predicted_angle = prediction["potentiometer"]["board_relative_angle_deg"]
                if predicted_angle is not None:
                    pot_errors.append(circular_error_degrees(predicted_angle, float(targets["pot_angle_deg"][i])))
                total += 1
    mean = lambda values: sum(values) / len(values) if values else None
    def classification_metrics(index: int) -> dict[str, float]:
        precision_denominator = switch_tp[index] + switch_fp[index]
        recall_denominator = switch_tp[index] + switch_fn[index]
        precision = switch_tp[index] / precision_denominator if precision_denominator else 0.0
        recall = switch_tp[index] / recall_denominator if recall_denominator else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        return {
            "accuracy": switch_correct[index] / switch_counts[index],
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    return {
        "images": total,
        "orientation_mae_deg": mean(orientation_errors),
        "switch_metrics": {"left": classification_metrics(0), "right": classification_metrics(1)},
        "exact_switch_configuration_accuracy": exact_correct / total,
        "pot_center_mean_error_px": mean(point_errors["center"]),
        "pot_tip_mean_error_px": mean(point_errors["tip"]),
        "pot_angle_mae_deg": mean(pot_errors),
        "pot_angle_valid_images": len(pot_errors),
        "pot_angle_undefined_images": total - len(pot_errors),
    }
