# openpi pi0.5 SONIC tactile training

Three fixed configs share the same code, dataset, action space, and normalization statistics:

| Config | Current tactile | Future tactile | Future state | Future stereo |
|---|---:|---:|---:|---:|
| `pi05_sonic_notactile` | no | no | no | no |
| `pi05_sonic_htd` | yes | yes | no | no |
| `pi05_sonic_jepa` (UniVLaT/JEPA) | yes | yes | yes | yes |

Future observations are stop-gradient auxiliary targets and never enter policy conditioning.
The legacy, CLI-overridable `pi05_sonic` config remains an alias of the no-tactile baseline.
Episode-tail samples are excluded whenever the 40-step action or auxiliary target window
would cross the episode boundary; repeated padding is never trained as a real target.

HTD is short for *Humanoid Transformer with Touch Dreaming* (arXiv:2604.13015). Here, HTD
mode names its current-tactile fusion and future-tactile latent objective; it does not imply
that pi0.5 reproduces the paper's complete policy and controller system.

## Environment and data assets

```bash
cd /home/wzh/Projects/Uni_VLaT/openpi
uv sync
export HF_LEROBOT_HOME=/home/wzh/Projects/Uni_VLaT/data

uv run python scripts/make_sonic_episodes_stats.py \
  --dataset-path /home/wzh/Projects/Uni_VLaT/data/desk_sweep
uv run python scripts/make_sonic_norm_stats.py \
  --dataset-path /home/wzh/Projects/Uni_VLaT/data/desk_sweep
```

Normalization is written once to
`assets/pi05_sonic/desk_sweep/norm_stats.json` and is shared by all three configs.

## Full training

The tested four-A800 configuration uses global batch 64 with four-way FSDP. Each command
runs 50k steps, or 3.2M samples, matching the completed Isaac-GR00T run (`50,000 x 64`).
The official LeRobot loader uses TorchCodec in this environment; a 120-step HTD run showed
no data starvation, so an additional decoded-video cache is not needed here.

```bash
cd /home/wzh/Projects/Uni_VLaT/openpi
export HF_LEROBOT_HOME=/home/wzh/Projects/Uni_VLaT/data
export CUDA_VISIBLE_DEVICES=0,1,2,3

COMMON_ARGS=(
  --exp-name train
  --num-train-steps 50000
  --save-interval 10000
  --keep-period 10000
  --batch-size 64
  --num-workers 4
  --fsdp-devices 4
)

uv run python scripts/train.py pi05_sonic_notactile "${COMMON_ARGS[@]}"
uv run python scripts/train.py pi05_sonic_htd "${COMMON_ARGS[@]}"
uv run python scripts/train.py pi05_sonic_jepa "${COMMON_ARGS[@]}"
```

Checkpoints are stored under `checkpoints/<config>/train/`; the final checkpoint is
`49999`. The checkpoint records its tactile graph switches in
`assets/jepa_model_config.json`, so serving restores the correct mode.

## Model server and SONIC bridge

Start the openpi websocket backend with the config matching the checkpoint:

```bash
cd /home/wzh/Projects/Uni_VLaT/openpi
export HF_LEROBOT_HOME=/home/wzh/Projects/Uni_VLaT/data
uv run python scripts/serve_policy.py --port 8000 \
  policy:checkpoint \
  --policy.config pi05_sonic_jepa \
  --policy.dir checkpoints/pi05_sonic_jepa/train/49999
```

Install the lightweight websocket client once and expose the backend through the GR00T ZMQ
interface expected by the shared controller:

```bash
cd /home/wzh/Projects/Uni_VLaT/Isaac-GR00T
uv pip install --python .venv/bin/python -e /home/wzh/Projects/Uni_VLaT/openpi/packages/openpi-client
uv run --no-sync python -m gr00t.eval.run_openpi_bridge_server \
  --openpi-host 127.0.0.1 --openpi-port 8000 --port 5550
```

Run the existing SONIC launcher without backend-specific changes:

```bash
cd /home/wzh/Projects/Uni_VLaT/GR00T-WholeBodyControl
python gear_sonic/scripts/launch_inference.py \
  --policy-host 127.0.0.1 --policy-port 5550 \
  --camera-host 192.168.123.164 --tactile-zmq-host 192.168.123.164 \
  --prompt "carry the bucket"
```

For `pi05_sonic_notactile`, omit `--tactile-zmq-host` and add `--no-use-tactile`.

## `sonic_vla_v1` contract

The websocket request contains `state: float32[46]`,
`ego_view_left/right: uint8[H,W,3]`, `prompt: str`, and tactile `uint8[768]` only for HTD/JEPA.
The response is finite `actions: float32[40,78]`, laid out as
`motion_token[0:64] | left_hand[64:71] | right_hand[71:78]`. The bridge validates these
dimensions and metadata before forwarding anything to the controller.
