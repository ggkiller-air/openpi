"""Generate the `meta/episodes_stats.jsonl` that lerobot v2.1 requires.

GR00T-format SONIC datasets ship an aggregate `meta/stats.json` but not the per-episode
`meta/episodes_stats.jsonl` that lerobot 0.1.0 demands when loading a v2.1 dataset. Without
it, lerobot falls back to the HF hub and fails (the dataset is local-only).

This replicates the aggregate stats for every episode (lerobot's own
`backward_compatible_episodes_stats` behavior) and adds a `count` field so the stats pass
lerobot's `aggregate_stats` validation. openpi recomputes its own norm stats and reads raw
data, so these values only need to be schema-valid — they do not affect training accuracy.

Idempotent: refuses to overwrite an existing file unless `--force`.

Usage:
    uv run python scripts/make_sonic_episodes_stats.py --dataset-path /data/zihao/Isaac-GR00T/data/carry-bucket-stereo
"""

import dataclasses
import json
import pathlib

import tyro


@dataclasses.dataclass
class Args:
    # Path to the LeRobot dataset directory (the one containing `meta/`).
    dataset_path: str
    # Overwrite meta/episodes_stats.jsonl if it already exists.
    force: bool = False


def main(args: Args) -> None:
    root = pathlib.Path(args.dataset_path)
    meta = root / "meta"
    stats_path = meta / "stats.json"
    episodes_path = meta / "episodes.jsonl"
    out_path = meta / "episodes_stats.jsonl"

    if not stats_path.exists():
        raise FileNotFoundError(f"{stats_path} not found — is --dataset-path a LeRobot dataset dir?")
    if not episodes_path.exists():
        raise FileNotFoundError(f"{episodes_path} not found — is --dataset-path a LeRobot dataset dir?")
    if out_path.exists() and not args.force:
        print(f"✓ {out_path} already exists, nothing to do (use --force to regenerate).")
        return

    stats = json.loads(stats_path.read_text())
    episodes = [json.loads(line)["episode_index"] for line in episodes_path.read_text().splitlines() if line.strip()]

    # Add a count field (shape (1,)) to every feature so lerobot's weighted aggregation works.
    # Identical across episodes -> re-aggregation reproduces the same mean/std.
    stats_with_count = {fk: {**fv, "count": [1]} for fk, fv in stats.items()}

    with open(out_path, "w") as f:
        for ep in episodes:
            f.write(json.dumps({"episode_index": ep, "stats": stats_with_count}) + "\n")

    print(f"✓ wrote {out_path}  ({len(episodes)} episodes, {len(stats_with_count)} features each)")


if __name__ == "__main__":
    main(tyro.cli(Args))
