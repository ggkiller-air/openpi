"""Build openpi normalization stats from a GR00T SONIC dataset's aggregate stats.

The generic ``compute_norm_stats.py`` decodes every camera frame even though normalization
only uses state and actions. SONIC's input transform only reorders state dimensions and
concatenates action columns, so the same aggregate statistics can be transformed directly.
"""

import dataclasses
import json
import pathlib

import numpy as np
import tyro

from openpi.policies import sonic_policy
from openpi.shared import normalize


@dataclasses.dataclass
class Args:
    dataset_path: pathlib.Path
    output_dir: pathlib.Path | None = None


def _feature_stats(raw_stats: dict, key: str, start: int | None = None, end: int | None = None) -> dict:
    if key not in raw_stats:
        raise KeyError(f"{key!r} is missing from meta/stats_gr00t.json")
    result = {}
    for field in ("mean", "std", "q01", "q99"):
        values = np.asarray(raw_stats[key][field], dtype=np.float32)
        result[field] = values[slice(start, end)]
    return result


def _concatenate(parts: list[dict]) -> normalize.NormStats:
    return normalize.NormStats(
        mean=np.concatenate([part["mean"] for part in parts]),
        std=np.concatenate([part["std"] for part in parts]),
        q01=np.concatenate([part["q01"] for part in parts]),
        q99=np.concatenate([part["q99"] for part in parts]),
    )


def build_sonic_norm_stats(dataset_path: pathlib.Path) -> dict[str, normalize.NormStats]:
    meta = dataset_path / "meta"
    stats_path = meta / "stats_gr00t.json"
    modality_path = meta / "modality.json"
    if not stats_path.exists() or not modality_path.exists():
        raise FileNotFoundError(f"expected {stats_path} and {modality_path}")

    raw_stats = json.loads(stats_path.read_text())["statistics"]
    modality = json.loads(modality_path.read_text())

    state_parts = []
    for group in sonic_policy.STATE_GROUP_ORDER:
        spec = modality["state"][group]
        state_parts.append(
            _feature_stats(
                raw_stats,
                spec.get("original_key", "observation.state"),
                int(spec["start"]),
                int(spec["end"]),
            )
        )
    gravity_spec = modality["state"]["projected_gravity"]
    state_parts.append(
        _feature_stats(
            raw_stats,
            gravity_spec.get("original_key", "observation.projected_gravity"),
            int(gravity_spec["start"]),
            int(gravity_spec["end"]),
        )
    )

    action_parts = []
    for group in ("motion_token", "left_hand_joints", "right_hand_joints"):
        spec = modality["action"][group]
        action_parts.append(_feature_stats(raw_stats, spec["original_key"], int(spec["start"]), int(spec["end"])))

    result = {"state": _concatenate(state_parts), "actions": _concatenate(action_parts)}
    if result["state"].mean.shape != (sonic_policy.SONIC_STATE_DIM,):
        raise ValueError(f"assembled state stats have shape {result['state'].mean.shape}, expected (46,)")
    if result["actions"].mean.shape != (sonic_policy.SONIC_ACTION_DIM,):
        raise ValueError(f"assembled action stats have shape {result['actions'].mean.shape}, expected (78,)")
    return result


def main(args: Args) -> None:
    output_dir = args.output_dir or pathlib.Path("assets/pi05_sonic") / args.dataset_path.name
    normalize.save(output_dir, build_sonic_norm_stats(args.dataset_path))
    print(f"Wrote {output_dir / 'norm_stats.json'}")


if __name__ == "__main__":
    main(tyro.cli(Args))
