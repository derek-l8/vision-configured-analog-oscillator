from pathlib import Path

import torch
import pytest

from oscillator_cv.model import ModelConfig, MultiTaskCNN
from oscillator_cv.training import load_checkpoint, load_training_checkpoint, save_checkpoint, train_model


@pytest.mark.parametrize("normalization,grid", [("batch", 1), ("group", 4)])
def test_checkpoint_restores_model_optimizer_and_epoch(tmp_path: Path, normalization: str, grid: int) -> None:
    config = ModelConfig(base_channels=2, input_width=32, input_height=48,
                         normalization=normalization, switch_pool_grid=grid)
    model = MultiTaskCNN(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    image = torch.rand(2, 3, 48, 32)
    model(image)["switch_logits"].sum().backward()
    optimizer.step()
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, model, optimizer, 3, [{"epoch": 3.0}], "synthetic-smoke-fixtures")

    restored, restored_optimizer, epoch, history = load_training_checkpoint(path, torch.device("cpu"), 0.001)
    assert restored.config == config
    assert restored_optimizer.param_groups[0]["lr"] == 0.001
    assert epoch == 3
    assert history == [{"epoch": 3.0}]
    for expected, actual in zip(model.parameters(), restored.parameters()):
        assert torch.equal(expected, actual)
    assert restored_optimizer.state_dict()["state"]


class UnsupportedCheckpointValue:
    """An inert object that the restricted checkpoint loader must reject."""


def test_checkpoint_loader_rejects_serialized_objects(tmp_path: Path) -> None:
    path = tmp_path / "unsupported.pt"
    torch.save({"unsupported": UnsupportedCheckpointValue()}, path)
    with pytest.raises(ValueError, match="unsupported serialized objects"):
        load_checkpoint(path, torch.device("cpu"))


def test_training_seed_controls_model_fit(tmp_path: Path) -> None:
    from oscillator_cv.dataset import SyntheticOscillatorDataset

    dataset = SyntheticOscillatorDataset(samples=2, width=32, height=48, seed=19)
    config = ModelConfig(base_channels=2, input_width=32, input_height=48, normalization="group")
    states = []
    for index, seed in enumerate((23, 23, 24)):
        path = train_model(dataset, tmp_path / f"run-{index}", epochs=1, batch_size=2,
                           learning_rate=0.001, model_config=config, seed=seed)
        checkpoint = load_checkpoint(path, torch.device("cpu"))
        assert checkpoint["training_seed"] == seed
        states.append(checkpoint["model_state"])
    assert all(torch.equal(states[0][key], states[1][key]) for key in states[0])
    assert any(not torch.equal(states[0][key], states[2][key]) for key in states[0])


def test_resume_rejects_checkpoint_with_legacy_loss(tmp_path: Path) -> None:
    path = tmp_path / "legacy.pt"
    torch.save({"loss_version": "legacy-pixel-bce"}, path)
    with pytest.raises(ValueError, match="different heatmap loss"):
        load_training_checkpoint(path, torch.device("cpu"), 0.001)


def test_epoch_resume_rejects_step_based_overfit_checkpoint(tmp_path: Path) -> None:
    from oscillator_cv.training import LOSS_VERSION
    path = tmp_path / "diagnostic.pt"
    torch.save({"loss_version": LOSS_VERSION, "training_source": "annotated-images-overfit-diagnostic"}, path)
    with pytest.raises(ValueError, match="optimization steps"):
        load_training_checkpoint(path, torch.device("cpu"), 0.001)
