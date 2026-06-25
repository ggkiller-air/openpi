# SONIC 集成 —— 改动文件清单 + 代码参阅顺序

配合 `method.md`(讲思想)一起看。这份讲**改了哪些文件**,以及**按什么顺序读代码**才能从「不懂 π0.5」到「看懂这套改动」。

---

## Part 0. 先读这些 upstream 文件,建立 π0.5 基础

按顺序读,每个文件只需抓住括号里的重点,不用逐行:

| 顺序 | 文件 | 看什么 |
|---|---|---|
| 1 | `src/openpi/models/model.py` | `IMAGE_KEYS`(写死的 3 个图像槽位);`Observation` 数据结构(images / state / tokenized_prompt);`preprocess_observation` |
| 2 | `src/openpi/models/pi0_config.py` | `Pi0Config` 的字段:`action_dim`(默认 32)、`action_horizon`(默认 50)、`pi05`、`max_token_len`。这是模型的「形状」配置 |
| 3 | `src/openpi/models/pi0.py` | `__init__` 里的 `action_in_proj` / `action_out_proj`(绑 `action_dim` 的两层);`embed_prefix`(`for name in obs.images:` —— 所有图共用 `self.PaliGemma.img`,**槽位名无语义**);flow matching 的采样在 `sample_actions` |
| 4 | `src/openpi/policies/libero_policy.py` | **这是我们 sonic_policy 的模板**。看 `LiberoInputs`(怎么把数据集字段→模型输入:图像、state、actions、prompt)和 `LiberoOutputs`(怎么把模型输出切回机器人维度) |
| 5 | `src/openpi/training/config.py` 的 `LeRobotLiberoDataConfig`(约 282 行) | **这是我们 SonicDataConfig 的模板**。看 `repack_transforms`(数据集 key 改名)、`data_transforms`(Inputs/Outputs)、`model_transforms`(归一化+tokenize) 三段怎么拼 |
| 6 | `src/openpi/transforms.py` | `RepackTransform`(`{新key: 数据集原path}`,**只保留映射里的 key**)、`Normalize`/`Unnormalize`(注意 `std + 1e-6` 对 0 方差的保护)、`PadStatesAndActions`(把 state/action 补零到 `action_dim`) |
| 7 | `src/openpi/training/weight_loaders.py` 的 `CheckpointWeightLoader` + `_merge_params` | **这是我们自定义 loader 的基础**。看 `_merge_params(missing_regex=...)` 怎么「缺的 key 从 fresh 参数补回」 |
| 8 | `src/openpi/policies/policy_config.py` 的 `create_trained_policy` | 推理时 transform 链怎么拼(注意:**训练用的 repack 不在推理时跑**);`Policy.infer(obs) -> {"actions": ...}` |

读完这 8 个,你就懂 π0.5 的数据流:`数据集 → repack → Inputs → 归一化 → tokenize → 模型 → 反归一化 → Outputs`。

---

## Part 1. 我们改/加了哪些文件

### openpi 侧(3 个)

#### A. `src/openpi/policies/sonic_policy.py`(新增)
对照 `libero_policy.py` 看。三个东西:
- `SonicInputs`:把一条样本变成模型输入。
  - **state 装配**(`assemble_state_46`):按 `STATE_GROUP_ORDER`(=推理 modality_keys 顺序)从原始 43 维列切出各组、再拼成 46 维 + projected_gravity。每组在列里的位置(spans)**由 `SonicDataConfig` 从数据集 `meta/modality.json` 自动读取**(见 `config.py` 的 `_read_sonic_state_spans`),不手敲——任何 SONIC 数据集都自动对齐。`DEFAULT_STATE_SPANS` 只是读不到时的兜底。
  - **双路兼容**:训练时收到 `state_43`+`projected_gravity` 自己拼;推理时桥接直接发好的 `state`(46 维),直接用。这保证训练==推理同一条 46 维。
  - 图像:`ego_view_left→base_0_rgb`、`ego_view_right→left_wrist_0_rgb`、第三槽 zeros+mask False。
  - 动作:`concat([motion_token, left_hand_joints, right_hand_joints])` → 78 维。
- `SonicOutputs`:把模型输出切到前 78 维(去掉 pad)。
- 顶部常量 `MOTION_TOKEN_DIM=64 / LEFT_HAND_DIM=7 / RIGHT_HAND_DIM=7 / SONIC_ACTION_DIM=78` —— **桥接侧拆分动作时必须和这里一致**。

#### B. `src/openpi/training/weight_loaders.py`(加了 `PartialCheckpointWeightLoader`)
- 比 `CheckpointWeightLoader` 多做一件事:加载 base 后,**删掉**①命中 `skip_regex`(`action_in_proj|action_out_proj|state_proj`)的 key ②shape 和 fresh 不一致的 key;再用 `_merge_params` 把这些 key 从 fresh 参数补回(=保持随机初始化)。
- 效果:主干从 base 加载,动作头重新初始化。训练第 0 步若维度不对会立刻报错。

#### C. `src/openpi/training/config.py`(加了 `SonicDataConfig` + `pi05_sonic`)
- `SonicDataConfig`:对照 `LeRobotLiberoDataConfig`。
  - `action_sequence_keys = (action.motion_token, teleop.left_hand_joints, teleop.right_hand_joints)` —— 这三列各按 horizon=40 读成序列。
  - `repack`:把数据集原始列名映射到 `SonicInputs` 用的短名(**含 `"prompt":"prompt"`,否则 prompt 会被丢**)。
  - 不加 Delta/Absolute(SONIC 动作是绝对值)。
- `pi05_sonic` 这个 `TrainConfig`:`Pi0Config(pi05=True, action_dim=78, action_horizon=40)`、`weight_loader=PartialCheckpointWeightLoader(pi05_base)`、`repo_id="carry-bucket-stereo"`。

### GR00T 侧(2 个)

#### D. `gr00t/policy/openpi_bridge_policy.py`(新增)
- `OpenpiBridgePolicy(BasePolicy)`:GR00T 的策略接口,内部是一个 openpi websocket **客户端**。
- `_get_action(observation)`:
  - 按 modality config 的 8 个 state key **顺序**拼 46 维(和 `SonicInputs` 推理契约一致)。
  - 取两路图、prompt,发给 openpi `client.infer(...)`。
  - 拿到 `[40,78]`,**拆**成 `motion_token[:64] / left_hand_joints[64:71] / right_hand_joints[71:78]`,stack 成 `(B,40,D)`、转 float32 返回。

#### E. `gr00t/eval/run_openpi_bridge_server.py`(新增)
- 把 `OpenpiBridgePolicy` 挂到 GR00T 的 `PolicyServer`(默认 5550)。旧 sonic 客户端连这里,协议不变。



---

## Part 2. 两条数据路径(对照着记)

**训练时**(在 openpi 里跑):
```
LeRobot 数据集
 → PromptFromLeRobotTask 注入 prompt
 → repack(改名,只留映射的 key)           ← SonicDataConfig
 → SonicInputs(state_43+grav→46重排, 3动作列→78拼接)
 → Normalize(用 compute_norm_stats 的统计量)
 → tokenize + PadStatesAndActions(补到78)
 → π0.5 训练
```

**推理时**(桥接 → openpi serve):
```
GR00T 客户端 ──ZeroMQ──► 5550 桥接服务
 → OpenpiBridgePolicy 拼 46 维 state、取图、取 prompt
 ──websocket──► openpi serve:8000
   → (推理链没有 repack) InjectDefaultPrompt → SonicInputs(走"state"分支,直接用46维)
   → Normalize → tokenize → π0.5.sample → 反归一化 → SonicOutputs(切78)
 ◄── 返回 [40,78]
 → 桥接拆成 motion_token/left_hand/right_hand (B,40,D)
 ──ZeroMQ──► 回 GR00T 客户端
```

记住一个反差:**repack 只在训练时跑**;推理时桥接直接把 46 维 state 发过去,所以 `SonicInputs` 写了「有 `state` 就直接用,否则用 `state_43`+`grav` 拼」的双分支。

---

## Part 3. 验证过的点(可复现)

- 配置导入、`get_config("pi05_sonic")` 形状正确(action_dim=78, horizon=40)。
- `SonicInputs` 单测:46 维重排顺序正确、78 维动作拼接正确、图像槽位+mask 正确、训练/推理双路径都通。
- `PartialCheckpointWeightLoader` 合成参数测试:主干加载、`action_in/out_proj` 重初始化、key 集完整。
- 完整数据管线端到端:`state→78`、`actions (40,78)`、3 图像槽位、`tokenized_prompt` 存在。
- 桥接 `_get_action` 单测(mock client):动作拆分偏移正确、46 维 state 装配、prompt 提取。
- `compute_norm_stats` 跑通,`motion_token` 方差健康;手部维度恒定(本版本采集时未用手,正常)被 `+1e-6` 保护。

> 唯一没在本地跑的是「多小时正式训练」和「真机端到端」——你自己跑。训练第 0 步能过就说明 weight loader 正确。

---

## Part 5. 触觉模态(HTD)—— 改了哪些文件

思路见 `method.md` 第六节。开关:`--model.use-tactile`(总开关)+ `--model.tactile-encoder-type mlp|cnn|coord`。**关掉 = baseline 字节级不变**。

### 数据流(触觉,训练时)
```
observation.tactile_raw [256] uint8
 → data_loader 按 [0..4] 开窗 → [5,256]            ← DataConfig.tactile_key
 → repack "tactile" → SonicInputs(原样 uint8 透传, 不归一化)
 → Observation.tactile [B,5,256]
 → embed_suffix: tactile_encoder(tactile[:,0]) → 8 token 注入 action token 前
 → compute_loss: dream_head(trunk[8位]) vs teacher.encode_pooled(tactile[:,1:5]) → L_dream
 → train.py: loss = action_loss + λ·L_dream;teacher 每步 EMA
```
推理时只有当前帧:桥接发 [256] → SonicInputs 包成 [1,256] → 编码器吃 `tactile[:,0]`,teacher/dream 不参与。

### 新增文件
- **`src/openpi/models/tactile.py`**:`TactileEncoder`(valid_idx 选 112 + /255 + per-region encoder + `SlotAggregator`)、`DreamHead`、`dream_loss`。`VALID_IDX/REGION_SIZES/GRIDS` 复用 GR00T `tactile_layout.py`。embed=1024。三种 per-region 编码器都实现:`PerRegionMLP`(mlp)、`PerRegionCNN`(cnn / coord)。
  - **CNN 从 PyTorch 移到 JAX 的三个差异(已验证)**:① `nnx.Conv` 是 **NHWC** → region reshape `[B,r,c,1]`(PyTorch 是 NCHW `[B,1,r,c]`,内容 row-major 一致);② JAX 无 `AdaptiveAvgPool2d` → `_adaptive_avg_pool2d` 用平均矩阵 `Mh/Mw` 自实现,**逐元素对齐 PyTorch**(含 5×8→2×2 不整除);③ coord 通道拼最后轴、`coord_scale=0.1`(gotcha #3)。param 数 cnn(4.87M)< mlp(7.42M),符合规格"conv 权重共享更省参"。

### 改动文件(7 个)
| 文件 | 改了什么 |
|---|---|
| `models/model.py` | `Observation` 加 `tactile` 字段;`from_dict`/`preprocess_observation` 原样透传(**不归一化、不增广**) |
| `models/pi0_config.py` | `Pi0Config` 加 `use_tactile/tactile_encoder_type/dream_horizon`;`inputs_spec` 开时加 `tactile [B,τ+1,256] uint8`(供 `fake_obs` 初始化) |
| `models/pi0.py` | `__init__` 建 `tactile_encoder/tactile_teacher/dream_head`(teacher = student 副本);`embed_suffix` 注入 8 token(ar_mask `[True]+[False]*7`,prefix 看不到、action 看得到;cast 到 action token dtype);`compute_loss` 开时返回 `(chunked, dream_aux)`;`sample_actions` 走同一 `embed_suffix` 自动带触觉 |
| `scripts/train.py` | `loss_fn` 兼容元组/数组返回,`loss=action+λ·dream`,`has_aux=True`;主 EMA 块后加 **teacher EMA**(`nnx.state(encoder/teacher)` → `jax.tree.map` 混合 → `nnx.update`);`info` 加 `action_loss/dream_loss` |
| `training/config.py` | `TrainConfig.lambda_tactile=0.5 / tactile_ema_decay=0.99`;`trainable_filter` **永久排除** `tactile_teacher`;`DataConfig.tactile_key/tactile_horizon`;`SonicDataConfig` 按 `model.use_tactile` 自动设 tactile_key + repack 加 `"tactile"` |
| `training/data_loader.py` | `create_torch_dataset`:`tactile_key` 非空时把它并入 `delta_timestamps`(`range(tactile_horizon)`)→ 触觉开窗 |
| `policies/sonic_policy.py` | `SonicInputs` 加 `tactile` 分支(uint8 原样;[256]→[1,256]) |
| `Isaac-GR00T/gr00t/policy/openpi_bridge_policy.py` | `_get_action` 若 obs 含 tactile,转发当前帧 `tactile_raw[i,0]` 给 openpi |

### 参阅顺序(看懂触觉这套)
1. `method.md` 第六节(思路)→ 2. `src/openpi/models/tactile.py`(编码器/dream 全在这)→ 3. `pi0.py` 的 `embed_suffix`(注入)+ `compute_loss`(dream 损失)→ 4. `scripts/train.py` 的 `loss_fn` + teacher EMA 块 → 5. `config.py` 的 `SonicDataConfig` + `trainable_filter` + `data_loader.py` 开窗。

### 验证过的点(CPU,可复现)
- `tactile.py`:三种编码器 `forward[2,256]→[2,8,1024]`、`encode_pooled[2,5,256]→[2,5,1024]`、`dream_loss` 自身≈0。
- `_adaptive_avg_pool2d` 对 5 种分区(含 5×8→2×2 不整除)**逐元素等于 torch `AdaptiveAvgPool2d`**。
- 模型(mlp/cnn/coord 三种):`use_tactile=True` → `compute_loss` 返回 `(chunked[1,40], aux 有限)`;`False` → 返回数组(baseline 不变)。
- 数据管线:`tactile [5,256] uint8` 未归一化到达模型;动作/状态仍 78。
- **teacher EMA**(三种编码器):teacher 全排除出 trainable;init 时 == student;EMA 一步精确移动 0.01。
- 全部改动文件 `py_compile` + 导入通过;桥接导入通过。

### 待办 / 风险
- **GPU 训练 + 真机未验证**(本轮无卡):第 0 步 `dream_loss` 应有限、loss 下降;loss 突然 0.0 多半是 NaN(规格 gotcha #2)。三种编码器都只在 CPU 形状/前向验证过。
- **部署**:GR00T 端 modality config 必须含 tactile,桥接才转发得到。


