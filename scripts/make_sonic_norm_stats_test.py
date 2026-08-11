import json

import numpy as np

from scripts import make_sonic_norm_stats


def _stats(values):
    values = np.asarray(values, dtype=np.float32)
    return {
        "mean": values.tolist(),
        "std": (values + 100).tolist(),
        "q01": (values + 200).tolist(),
        "q99": (values + 300).tolist(),
    }


def test_build_sonic_norm_stats_reorders_state_and_concatenates_actions(tmp_path):
    meta = tmp_path / "meta"
    meta.mkdir()
    raw_state = np.arange(43, dtype=np.float32)
    raw_stats = {
        "observation.state": _stats(raw_state),
        "observation.projected_gravity": _stats(np.arange(3) + 1000),
        "action.motion_token": _stats(np.arange(64) + 2000),
        "teleop.left_hand_joints": _stats(np.arange(7) + 3000),
        "teleop.right_hand_joints": _stats(np.arange(7) + 4000),
    }
    modality = {
        "state": {
            "left_leg": {"start": 0, "end": 6},
            "right_leg": {"start": 6, "end": 12},
            "waist": {"start": 12, "end": 15},
            "left_arm": {"start": 15, "end": 22},
            "left_hand": {"start": 22, "end": 29},
            "right_arm": {"start": 29, "end": 36},
            "right_hand": {"start": 36, "end": 43},
            "projected_gravity": {
                "start": 0,
                "end": 3,
                "original_key": "observation.projected_gravity",
            },
        },
        "action": {
            "motion_token": {"start": 0, "end": 64, "original_key": "action.motion_token"},
            "left_hand_joints": {
                "start": 0,
                "end": 7,
                "original_key": "teleop.left_hand_joints",
            },
            "right_hand_joints": {
                "start": 0,
                "end": 7,
                "original_key": "teleop.right_hand_joints",
            },
        },
    }
    (meta / "stats_gr00t.json").write_text(json.dumps({"statistics": raw_stats}))
    (meta / "modality.json").write_text(json.dumps(modality))

    result = make_sonic_norm_stats.build_sonic_norm_stats(tmp_path)

    expected_state = np.concatenate(
        [raw_state[:22], raw_state[29:36], raw_state[22:29], raw_state[36:43], np.arange(3) + 1000]
    )
    expected_actions = np.concatenate([np.arange(64) + 2000, np.arange(7) + 3000, np.arange(7) + 4000])
    np.testing.assert_array_equal(result["state"].mean, expected_state)
    np.testing.assert_array_equal(result["actions"].mean, expected_actions)
    np.testing.assert_array_equal(result["state"].q99, expected_state + 300)
