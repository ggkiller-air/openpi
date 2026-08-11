# ruff: noqa: I001

import dataclasses

import jax

from openpi.models import pi0_config
from openpi.training import data_loader as _data_loader
from openpi.training import config as _config


def test_torch_data_loader():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 16)

    loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=4,
        num_batches=2,
    )
    batches = list(loader)

    assert len(batches) == 2
    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_torch_data_loader_infinite():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 4)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4)
    data_iter = iter(loader)

    for _ in range(10):
        _ = next(data_iter)


def test_torch_data_loader_parallel():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 10)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4, num_batches=2, num_workers=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_with_fake_dataset():
    config = _config.get_config("debug")

    loader = _data_loader.create_data_loader(config, skip_norm_stats=True, num_batches=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == config.batch_size for x in jax.tree.leaves(batch))

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)


def test_with_real_dataset():
    config = _config.get_config("pi0_aloha_sim")
    config = dataclasses.replace(config, batch_size=4)

    loader = _data_loader.create_data_loader(
        config,
        # Skip since we may not have the data available.
        skip_norm_stats=True,
        num_batches=2,
        shuffle=True,
    )
    # Make sure that we can get the data config.
    assert loader.data_config().repo_id == config.data.repo_id

    batches = list(loader)

    assert len(batches) == 2

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)


def test_jepa_delta_timestamps(monkeypatch):
    captured = {}

    class FakeMetadata:
        fps = 50

        def __init__(self, repo_id):
            del repo_id

    class FakeLeRobotDataset:
        def __init__(self, repo_id, *, delta_timestamps):
            captured["repo_id"] = repo_id
            captured["delta_timestamps"] = delta_timestamps

    monkeypatch.setattr(_data_loader.lerobot_dataset, "LeRobotDatasetMetadata", FakeMetadata)
    monkeypatch.setattr(_data_loader.lerobot_dataset, "LeRobotDataset", FakeLeRobotDataset)
    data_config = _config.DataConfig(
        repo_id="local/test",
        action_sequence_keys=("action",),
        tactile_keys=("tactile",),
        tactile_horizon=5,
        state_sequence_keys=("state", "gravity"),
        state_horizon=5,
        vision_sequence_keys=("left", "right"),
        vision_horizon=3,
    )

    _data_loader.create_torch_dataset(data_config, action_horizon=4, model_config=pi0_config.Pi0Config())

    assert captured["delta_timestamps"]["action"] == [0.0, 0.02, 0.04, 0.06]
    assert captured["delta_timestamps"]["tactile"] == [0.0, 0.02, 0.04, 0.06, 0.08]
    assert captured["delta_timestamps"]["state"] == [0.0, 0.02, 0.04, 0.06, 0.08]
    assert captured["delta_timestamps"]["gravity"] == [0.0, 0.02, 0.04, 0.06, 0.08]
    assert captured["delta_timestamps"]["left"] == [0.0, 0.02, 0.04]
    assert captured["delta_timestamps"]["right"] == [0.0, 0.02, 0.04]
