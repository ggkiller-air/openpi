from types import SimpleNamespace

import numpy as np
import pytest

from openpi.models import model as _model
from openpi.policies import sonic_policy


def test_assemble_state_46_supports_time_windows_and_canonical_order():
    state = np.arange(2 * 43, dtype=np.float32).reshape(2, 43)
    gravity = np.arange(6, dtype=np.float32).reshape(2, 3) + 1000

    result = sonic_policy.assemble_state_46(state, gravity)
    expected = np.concatenate(
        [state[..., :22], state[..., 29:36], state[..., 22:29], state[..., 36:43], gravity], axis=-1
    )

    assert result.shape == (2, sonic_policy.SONIC_STATE_DIM)
    np.testing.assert_array_equal(result, expected)


def test_sonic_inputs_split_current_and_future_stereo_frames():
    horizon = 4
    left = np.stack([np.full((6, 8, 3), value, np.uint8) for value in range(horizon + 1)])
    right_thwc = np.stack([np.full((6, 8, 3), 10 + value, np.uint8) for value in range(horizon + 1)])
    right_tchw = right_thwc.transpose(0, 3, 1, 2)
    state = np.arange((horizon + 1) * 43, dtype=np.float32).reshape(horizon + 1, 43)
    gravity = np.ones((horizon + 1, 3), dtype=np.float32)

    transform = sonic_policy.SonicInputs(
        model_type=_model.ModelType.PI05,
        dream_state=True,
        dream_vision=True,
        vision_horizon=horizon,
    )
    output = transform(
        {
            "state_43": state,
            "projected_gravity": gravity,
            "ego_view_left": left,
            "ego_view_right": right_tchw,
        }
    )

    assert output["state"].shape == (horizon + 1, sonic_policy.SONIC_STATE_DIM)
    assert output["image"]["base_0_rgb"].shape == (6, 8, 3)
    assert output["image"]["left_wrist_0_rgb"].shape == (6, 8, 3)
    np.testing.assert_array_equal(output["image"]["base_0_rgb"], left[0])
    np.testing.assert_array_equal(output["image"]["left_wrist_0_rgb"], right_thwc[0])
    assert output["future_images"]["base_0_rgb"].shape == (horizon, 6, 8, 3)
    assert output["future_images"]["left_wrist_0_rgb"].shape == (horizon, 6, 8, 3)
    np.testing.assert_array_equal(output["future_images"]["base_0_rgb"], left[1:])
    np.testing.assert_array_equal(output["future_images"]["left_wrist_0_rgb"], right_thwc[1:])


def test_sonic_dream_inference_accepts_current_only_observation():
    transform = sonic_policy.SonicInputs(
        model_type=_model.ModelType.PI05,
        dream_state=True,
        dream_vision=True,
        vision_horizon=4,
    )
    output = transform(sonic_policy.make_sonic_example())

    assert output["state"].shape == (1, sonic_policy.SONIC_STATE_DIM)
    assert output["image"]["base_0_rgb"].ndim == 3
    assert "future_images" not in output


def test_sonic_metadata_is_derived_from_restored_checkpoint_config():
    metadata = sonic_policy.make_sonic_metadata(
        SimpleNamespace(action_dim=78, action_horizon=40, use_tactile=True)
    )

    assert metadata["protocol"] == "sonic_vla_v1"
    assert metadata["requires_tactile"] is True
    assert metadata["video_keys"] == ["ego_view_left", "ego_view_right"]


def test_sonic_tactile_checkpoint_rejects_missing_or_malformed_tactile():
    transform = sonic_policy.SonicInputs(
        model_type=_model.ModelType.PI05,
        requires_tactile=True,
    )
    data = sonic_policy.make_sonic_example()
    with pytest.raises(ValueError, match="requires a current tactile"):
        transform(data)

    data["tactile"] = np.zeros(767, dtype=np.uint8)
    with pytest.raises(ValueError, match="shape"):
        transform(data)


def test_sonic_tactile_accepts_lossless_lerobot_integer_cast_only():
    transform = sonic_policy.SonicInputs(
        model_type=_model.ModelType.PI05,
        requires_tactile=True,
    )
    data = sonic_policy.make_sonic_example()
    data["tactile"] = np.arange(768, dtype=np.int64) % 256
    output = transform(data)
    assert output["tactile"].dtype == np.uint8

    data["tactile"][-1] = 256
    with pytest.raises(ValueError, match="uint8-compatible"):
        transform(data)

    data["tactile"] = np.zeros(768, dtype=np.float32)
    with pytest.raises(ValueError, match="uint8-compatible"):
        transform(data)


def test_sonic_outputs_reject_nonfinite_or_wrong_horizon():
    transform = sonic_policy.SonicOutputs()
    with pytest.raises(ValueError, match="shape"):
        transform({"actions": np.zeros((39, 78), dtype=np.float32)})

    actions = np.zeros((40, 78), dtype=np.float32)
    actions[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        transform({"actions": actions})


def test_sonic_inputs_fail_on_incomplete_future_window():
    transform = sonic_policy.SonicInputs(
        model_type=_model.ModelType.PI05,
        dream_vision=True,
        vision_horizon=4,
    )
    data = sonic_policy.make_sonic_example()
    data["ego_view_left"] = np.zeros((4, 6, 8, 3), dtype=np.uint8)
    data["ego_view_right"] = np.zeros((4, 6, 8, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="expects 5 stereo frames"):
        transform(data)
