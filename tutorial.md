# 0. 一次性准备

## 0.1 补数据集缺失文件(每个新数据集跑一次)
GR00T 数据集缺 lerobot v2.1 要求的 `meta/episodes_stats.jsonl`,先用脚本补上(幂等,已生成会自动跳过):
```bash
uv run python scripts/make_sonic_episodes_stats.py \
    --dataset-path /data/zihao/Isaac-GR00T/data/carry-bucket-stereo
```
[scripts/make_sonic_episodes_stats.py](scripts/make_sonic_episodes_stats.py)



---

# 1. Training

## 1.1 换数据集时需要修改 `pi05_sonic` 里的 `repo_id`;state 顺序会自动从该数据集的 `modality.json` 读取。

  [src/openpi/training/config.py](src/openpi/training/config.py)
  
## 1.2 算归一化统计(每个数据集一次)
```bash
export HF_LEROBOT_HOME=/data/zihao/Isaac-GR00T/data
uv run python scripts/compute_norm_stats.py --config-name=pi05_sonic
# 写到 assets/pi05_sonic/<repo_id>/norm_stats.json
```

## 1.3 launch training
全量 fine-tune(所有权重都训,无 LoRA / 无冻结)。pi0.5 全量约需 ~70GB/卡 → 单张 80GB 即可放下;
卡数由 `CUDA_VISIBLE_DEVICES` 决定,默认数据并行(每卡一份完整模型、分摊 batch)。
```bash
tmux new -s sonic_ft

export HF_LEROBOT_HOME=/data/zihao/Isaac-GR00T/data
export CUDA_VISIBLE_DEVICES=0,1,2,3   # 想用几张就写几张(8×80GB 数据并行最快)

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run python scripts/train.py pi05_sonic \
    --exp-name=tactile_coord \
    --num-train-steps=20000 \
    --save-interval=10000 \
    --num-workers=16 \
    --model.use-tactile \
    --model.tactile-encoder-type coord \
    --overwrite \
# checkpoint 存到 checkpoints/pi05_sonic/sonic_v1/<step>
# 仅当单卡显存不够时(<70GB)才需要把模型切片: 加 --fsdp-devices <卡数>
```

---

# 2. Inference(双进程桥接)

> openpi 在自己的 venv 起 websocket 服务;GR00T 侧用桥接把它挂到 ZeroMQ PolicyServer:5550。
> 旧的 sonic 客户端连 5550,**一行不改**。

## 2.1 进程A — openpi 策略服务(openpi venv)
```bash
cd /data/zihao/openpi
export HF_LEROBOT_HOME=/data/zihao/Isaac-GR00T/data

uv run python scripts/serve_policy.py --port 8000 \
    policy:checkpoint \
    --policy.config pi05_sonic \
    --policy.dir checkpoints/pi05_sonic/sonic_v1/<step>
```

## 2.2 进程B — 桥接服务(GR00T venv)
- 桥接策略:[gr00t/policy/openpi_bridge_policy.py](../Isaac-GR00T/gr00t/policy/openpi_bridge_policy.py)
- 启动器:[gr00t/eval/run_openpi_bridge_server.py](../Isaac-GR00T/gr00t/eval/run_openpi_bridge_server.py)
```bash
cd /data/zihao/Isaac-GR00T
python -m gr00t.eval.run_openpi_bridge_server \
    --port 5550 \
    --openpi-host 127.0.0.1 --openpi-port 8000
```

## 2.3 进程C — sonic 客户端(不改)
```bash
python gear_sonic/scripts/launch_inference.py \
    --prompt "carry the bucket" \
    --camera-host 192.168.123.164
```

