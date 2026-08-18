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

The tested four-A800 configuration uses global batch 64 with four-way FSDP. The practical
fine-tuning budget is 20k steps (1.28M samples, about 14 dataset passes), which takes about
16.5-17 hours for HTD including compilation and four checkpoints. This is a time-budgeted
fine-tune, not a forced match to Isaac-GR00T's sample count. Resume from the last checkpoint
only when validation or robot success is still improving. The official LeRobot loader uses
TorchCodec in this environment; a 120-step HTD run showed no data starvation.

```bash
cd /home/wzh/Projects/Uni_VLaT/openpi
export HF_LEROBOT_HOME=/home/wzh/Projects/Uni_VLaT/data
export CUDA_VISIBLE_DEVICES=0,1,2,3

COMMON_ARGS=(
  --exp-name train
  --num-train-steps 20000
  --save-interval 5000
  --keep-period 5000
  --batch-size 64
  --num-workers 4
  --fsdp-devices 4
)

uv run python scripts/train.py pi05_sonic_notactile "${COMMON_ARGS[@]}"
uv run python scripts/train.py pi05_sonic_htd "${COMMON_ARGS[@]}"
uv run python scripts/train.py pi05_sonic_jepa "${COMMON_ARGS[@]}"
```

Checkpoints are stored under `checkpoints/<config>/train/`; the final checkpoint is
`19999`. The checkpoint records its tactile graph switches in
`assets/jepa_model_config.json`, so serving restores the correct mode.

## Model server and SONIC bridge

Start the openpi websocket backend with the config matching the checkpoint:

```bash
cd /home/wzh/Projects/Uni_VLaT/openpi
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.90 \
  .venv/bin/python scripts/serve_policy.py --port 8000 \
  policy:checkpoint \
  --policy.config pi05_sonic_htd \
  --policy.dir /home/shared/outputs/openpi/pi05_sonic_htd/sonic_htd_20260813_openpi_first_v2/best_model/10000
```

Expose the backend through the GR00T ZMQ interface expected by the shared controller. The
bridge has its own websocket client; installing `openpi-client` is not required.

```bash
cd /home/wzh/Projects/Uni_VLaT/Isaac-GR00T
.venv/bin/python gr00t/eval/run_openpi_bridge_server.py \
  --openpi-host 127.0.0.1 --openpi-port 8000 --port 5550
```

Run the existing SONIC launcher without backend-specific changes:

```bash
cd /home/wzh/Projects/Uni_VLaT/GR00T-WholeBodyControl
python gear_sonic/scripts/launch_inference.py \
  --policy-host 127.0.0.1 --policy-port 5550 \
  --policy-timeout-ms 60000 \
  --camera-host 192.168.123.164 --tactile-zmq-host 192.168.123.164 \
  --prompt "carry the bucket"
```

For `pi05_sonic_notactile`, omit `--tactile-zmq-host` and add `--no-use-tactile`.
The verified HTD checkpoint fits a 24 GB RTX 4090 with the 0.90 JAX memory fraction above;
the first request includes JIT compilation and can take roughly 25 seconds.

## `sonic_vla_v1` contract

The websocket request contains `state: float32[46]`,
`ego_view_left/right: uint8[H,W,3]`, `prompt: str`, and tactile `uint8[4,768]` for the
four-frame JEPA method. The shared bridge builds this rolling history from WBC's current packets.
The response is finite `actions: float32[40,78]`, laid out as
`motion_token[0:64] | left_hand[64:71] | right_hand[71:78]`. The bridge validates these
dimensions and metadata before forwarding anything to the controller.
