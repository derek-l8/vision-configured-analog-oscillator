import torch

from oscillator_cv.model import ModelConfig, MultiTaskCNN
from oscillator_cv.dataset import gaussian_heatmap
from oscillator_cv.training import heatmap_argmax_points, multitask_loss, spatial_heatmap_loss
import pytest


def test_model_output_shapes_and_finite_loss() -> None:
    model = MultiTaskCNN(ModelConfig(base_channels=4, input_width=64, input_height=96))
    image = torch.rand(2, 3, 96, 64)
    outputs = model(image)
    assert outputs["orientation_sin_cos"].shape == (2, 2)
    assert outputs["switch_logits"].shape == (2, 2)
    assert outputs["heatmap_logits"].shape == (2, 2, 48, 32)
    assert torch.linalg.vector_norm(outputs["orientation_sin_cos"], dim=1).allclose(torch.ones(2), atol=1e-5)
    targets = {
        "orientation_sin_cos": torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
        "switch_states": torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
        "heatmaps": torch.rand(2, 2, 48, 32),
    }
    loss, components = multitask_loss(outputs, targets)
    assert torch.isfinite(loss)
    assert set(components) == {"total", "orientation", "switch", "heatmap"}


def test_spatial_loss_rewards_the_correct_location_not_background_suppression() -> None:
    target = gaussian_heatmap(24, 32, (8.25, 14.5)).view(1, 1, 32, 24)
    blank = torch.zeros_like(target)
    suppressed = blank - 100
    matched = target.clamp_min(1e-12).log()
    shifted = matched.roll(9, dims=-1)
    assert spatial_heatmap_loss(suppressed, target) == pytest.approx(float(spatial_heatmap_loss(blank, target)))
    assert float(spatial_heatmap_loss(matched, target)) < 1e-5
    assert spatial_heatmap_loss(matched, target) < spatial_heatmap_loss(shifted, target)
    logits = blank.clone().requires_grad_()
    spatial_heatmap_loss(logits, target).backward()
    assert logits.grad[0, 0, 14, 8] < 0  # gradient descent raises the intended peak
    assert logits.grad[0, 0, 0, 0] > 0


def test_spatial_loss_rejects_empty_or_mismatched_targets() -> None:
    with pytest.raises(ValueError, match="keypoint"):
        spatial_heatmap_loss(torch.zeros(1, 2, 8, 8), torch.zeros(1, 2, 8, 8))
    with pytest.raises(ValueError, match="shapes"):
        spatial_heatmap_loss(torch.zeros(1, 2, 8, 8), torch.ones(1, 2, 4, 4))


def test_peak_decoder_retains_subpixel_coordinates_and_handles_edges() -> None:
    point = (8.25, 14.375)
    logits = gaussian_heatmap(24, 32, point).clamp_min(1e-20).log()[None, None]
    assert heatmap_argmax_points(logits)[0, 0].tolist() == pytest.approx([2*point[0], 2*point[1]], abs=1e-5)
    assert heatmap_argmax_points(torch.zeros(1, 1, 4, 4))[0, 0].tolist() == [0, 0]


def test_group_normalization_predictions_match_train_and_eval() -> None:
    model = MultiTaskCNN(ModelConfig(base_channels=4, input_width=64, input_height=96, normalization="group"))
    image = torch.rand(2, 3, 96, 64)
    model.train()
    training_outputs = model(image)
    model.eval()
    with torch.no_grad():
        inference_outputs = model(image)
    for key in training_outputs:
        assert torch.allclose(training_outputs[key], inference_outputs[key], atol=1e-6)


def test_switch_head_retains_spatial_grid() -> None:
    model = MultiTaskCNN(ModelConfig(base_channels=4, input_width=64, input_height=96,
                                    normalization="group", switch_pool_grid=4))
    assert model.switch_head[-1].in_features == 4*6*4*4
    assert model(torch.rand(2, 3, 96, 64))["switch_logits"].shape == (2, 2)


def test_evaluation_reports_undefined_angles_instead_of_hiding_them() -> None:
    from oscillator_cv.dataset import SyntheticOscillatorDataset
    from oscillator_cv.training import evaluate_model

    class FlatHeatmapModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.placeholder = torch.nn.Parameter(torch.zeros(1))

        def forward(self, images):
            count = images.shape[0]
            return {"orientation_sin_cos": torch.tensor([[0.0, 1.0]]).repeat(count, 1),
                    "switch_logits": torch.zeros(count, 2),
                    "heatmap_logits": torch.zeros(count, 2, images.shape[2]//2, images.shape[3]//2)}

    metrics = evaluate_model(FlatHeatmapModel(), SyntheticOscillatorDataset(samples=2, width=32, height=48))
    assert metrics["pot_angle_mae_deg"] is None
    assert metrics["pot_angle_valid_images"] == 0
    assert metrics["pot_angle_undefined_images"] == 2
