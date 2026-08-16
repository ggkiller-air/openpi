# ruff: noqa: I001

import pathlib

import numpy as np
import pytest
import torch

from openpi.training import data_loader as _data_loader
from openpi.training import config as _config


@pytest.mark.parametrize(
    ("name", "expected_flags"),
    [
        ("pi05_sonic_notactile", (False, False, False, False)),
        ("pi05_sonic_htd", (True, True, False, False)),
        ("pi05_sonic_jepa", (True, True, True, True)),
    ],
)
def test_fixed_sonic_modes(name, expected_flags):
    train_config = _config.get_config(name)
    model = train_config.model

    assert (model.use_tactile, model.use_tactile_dream, model.dream_state, model.dream_vision) == expected_flags
    assert model.action_dim == 78
    assert model.action_horizon == 40
    assert train_config.data.assets.assets_dir == "./assets/pi05_sonic"
    if name == "pi05_sonic_jepa":
        assert model.use_tactile_temporal
        assert model.tactile_history_length == 4
        assert model.use_delta_targets


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("pi05_sonic_notactile", ((), 1, (), 1, (), 1)),
        ("pi05_sonic_htd", (("observation.tactile_vest", "observation.tactile_left_arm", "observation.tactile_right_arm"), 5, (), 1, (), 1)),
        (
            "pi05_sonic_jepa",
            (
                ("observation.tactile_vest", "observation.tactile_left_arm", "observation.tactile_right_arm"),
                8,
                ("observation.state", "observation.projected_gravity"),
                5,
                ("observation.images.ego_view_left", "observation.images.ego_view_right"),
                5,
            ),
        ),
    ],
)
def test_fixed_sonic_modes_load_only_enabled_future_targets(name, expected, monkeypatch, tmp_path):
    train_config = _config.get_config(name)
    monkeypatch.setattr(_config, "_read_sonic_state_spans", lambda _: None)
    monkeypatch.setattr(
        _config.SonicDataConfig,
        "_load_norm_stats",
        lambda self, assets_dir, asset_id: None,
    )

    data = train_config.data.create(pathlib.Path(tmp_path), train_config.model)

    assert (
        data.tactile_keys,
        data.tactile_horizon,
        data.state_sequence_keys,
        data.state_horizon,
        data.vision_sequence_keys,
        data.vision_horizon,
    ) == expected
    assert data.drop_incomplete_sequences


def test_episode_safe_dataset_drops_incomplete_windows():
    class FakeLeRobotDataset:
        def __init__(self):
            self.delta_indices = {"actions": np.array([0, 1, 2]), "state": np.array([0])}
            self.episode_data_index = {
                "from": torch.tensor([0, 5]),
                "to": torch.tensor([5, 9]),
            }

        def __getitem__(self, index):
            return int(index)

        def __len__(self):
            return 9

    dataset = _data_loader.EpisodeSafeDataset(FakeLeRobotDataset())
    assert dataset.valid_indices.tolist() == [0, 1, 2, 5, 6]
    assert [dataset[index] for index in range(len(dataset))] == [0, 1, 2, 5, 6]


def test_legacy_sonic_config_is_preserved():
    legacy = _config.get_config("pi05_sonic")
    named = _config.get_config("pi05_sonic_notactile")

    assert legacy.model == named.model
    assert legacy.data.repo_id == named.data.repo_id
