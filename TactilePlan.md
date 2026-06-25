# 用 π0.5 驱动 Unitree G1 SONIC —— 方法与思想

这份文档讲**为什么这么做**和**整体流程**。具体改了哪些文件、按什么顺序读代码,见 `details.md`。

---

## 一、先搞懂 π0.5 是什么

π0.5 是一个 **VLA(Vision-Language-Action)模型**:输入「图像 + 机器人状态 + 一句语言指令」,输出「未来一小段时间的动作序列」。

```
   图像(多路相机) ┐
   机器人状态 state ├──►  π0.5  ──►  动作块 actions: [action_horizon, action_dim]
   语言 prompt     ┘
```

它内部由两部分组成:

1. **PaliGemma**(视觉-语言主干):一个 SigLIP 视觉编码器 + Gemma 语言模型。把每张图编码成一串 token,把指令编码成一串 token,拼在一起,形成对「当前场景 + 任务」的理解。
2. **Action Expert**(动作专家):一个小 transformer,在 PaliGemma 的理解之上,用 **flow matching(流匹配)** 生成连续动作。

几个你必须建立的概念:

- **动作块 / action chunking**:模型一次不只预测「下一步」,而是预测未来 `action_horizon` 步(比如 40 步)的动作序列。机器人执行其中一部分,再重新推理。
- **`action_dim`**:每一步动作是一个 `action_dim` 维的向量。base 模型默认 32 维。
- **`action_horizon`**:预测多少步。base 默认 50。
- **flow matching**:一种连续生成方法——从随机噪声出发,经过若干步「去噪」,逐渐变成一个合理的动作向量。**关键点:它生成的是连续向量,所以任何「连续的东西」都能当动作让它回归**,这正是后面 SONIC 的 motion_token 能行的原因。
- **三个固定图像槽位**:模型写死了 3 个图像输入位 `base_0_rgb / left_wrist_0_rgb / right_wrist_0_rgb`。**所有图共用同一个视觉编码器,槽位名对模型没有语义**——名字只是历史叫法,关键是训练和推理用同一映射。
- **归一化**:状态和动作在喂入模型前会按数据集统计量归一化(π0.5 用分位数归一化 q01/q99),输出再反归一化。每个数据集要先 `compute_norm_stats`。
- **finetune**:加载在 1 万小时机器人数据上预训练的 base 权重,在你自己的小数据集上继续训,让它适配你的机器人。

---

## 二、SONIC 这一侧的「契约」

GR00T 里的 `unitree_g1_sonic` 这个具身,规定了观测和动作的精确格式(SONIC 全身控制器 WBC 就按这个吃):

**观测(给模型的):**
- 双目头部相机 `ego_view_left` / `ego_view_right`
- 46 维状态:`left_leg(6) right_leg(6) waist(3) left_arm(7) right_arm(7) left_hand(7) right_hand(7) projected_gravity(3)`
- 一句任务描述

**动作(模型要吐的),horizon = 40 步,每步:**
- `motion_token`(**64 维,SONIC latent**)—— 这不是关节角,而是一个**学习出来的隐变量**,WBC 拿它解码成全身动作
- `left_hand_joints`(7)、`right_hand_joints`(7)
- → 每步 64+7+7 = **78 维**,一次输出 `[40, 78]`

---

## 三、为什么不能直接拷贝 ckpt

GR00T 的推理服务器用 HuggingFace `AutoModel.from_pretrained` 加载,是 **PyTorch/HF 格式 + GR00T 自己的网络结构**;openpi 是 **JAX/orbax 格式 + π0.5 结构**。两者:

- 权重格式不同(safetensors vs orbax PyTree)、网络结构不同(没有对应的层)
- 通信协议不同(GR00T 用 ZeroMQ,openpi 用 websocket)
- **动作空间不同**(SONIC 要 78 维 latent,base π0.5 是 32 维通用动作)

所以只能走「**先 finetune,再桥接**」。

---

## 四、整体 Pipeline

```
┌─ 1. 数据 ──────────────────────────────────────────────┐
│  GR00T 的 SONIC LeRobot 数据集(carry-bucket-stereo)    │
│  ── openpi 数据变换 ──►  state 重排成 46 维             │
│                          motion_token|hands 拼成 78 维  │
└────────────────────────────────────────────────────────┘
                          │
┌─ 2. Finetune ──────────────────────────────────────────┐
│  π0.5 base (32 维动作头)                                │
│  ── 改 action_dim=78, horizon=40 ──►                    │
│  主干(PaliGemma+expert)照常加载,                       │
│  动作头(action_in/out_proj)因维度变了 → 重新初始化     │
│  在 SONIC 数据上训练,学会输出 78 维 SONIC 动作          │
└────────────────────────────────────────────────────────┘
                          │
┌─ 3. 部署(双进程桥接)─────────────────────────────────┐
│  进程A(openpi venv): 把训好的策略起成 websocket 服务   │
│  进程B(GR00T venv):  一个轻量 BasePolicy 当 websocket  │
│      客户端,转换观测/动作格式,挂在 GR00T 的           │
│      PolicyServer:5550 上                               │
└────────────────────────────────────────────────────────┘
                          │
        旧的 sonic 客户端连 5550 —— 完全不改
```

---

## 五、三个核心的「为什么」

理解了这三点,就理解了这套改动的全部精髓:

### 1. 动作头重新初始化(action_dim 32 → 78)
先分清两件事:**层的形状** vs **层里的权重(数值)**。
- 我们**确实**把 `action_in_proj`/`action_out_proj` 设成了 78 维(`action_dim=78`)——形状没问题。
- 纠结的是这俩层**初始填什么数**。

这两层是连接「78 维动作」和「动作专家内部语言(宽度 W)」的**翻译器**:`action_in_proj` 是 `(78, W)` 矩阵,`action_out_proj` 是 `(W, 78)`。base 的对应层是 `(32, W)`/`(W, 32)`。
**为什么不能把 base 的权重塞进去**:① 形状不一致(32 行塞不进 78 行);② 就算硬凑也是垃圾——base 的第 i 维是「它自己那套通用动作」,我们的第 i 维是「SONIC motion_token」,含义不同,后 46 维 base 更没见过。

**做法**:写一个自定义 weight loader,对这两个翻译器**不加载 base 的旧权重、保持随机初始化**(openpi 自带 loader 遇形状不一致会直接报错);中间的「大脑」(视觉、语言、动作专家)形状不变,照常从 base 加载。
**思想**:迁移学习最标准的「换头(swap the head)」——保留通用的「感知+推理」大脑,只把绑死动作格式的输入/输出翻译器换新、从头在 SONIC 数据上学。

### 2. motion_token 当连续动作回归
SONIC 的 64 维 motion_token 是个学习出来的隐变量。因为 flow matching 本来就是回归连续向量,所以**直接把这 64 维当成动作的一部分让模型去拟合**,完全成立。WBC 负责把 latent 解码回全身动作,这一段我们不碰。

### 3. 46 维 state 的顺序训练==推理(已自动对齐,无需担心)
**注意:这不是采集错误。** 同一份状态在 GR00T 体系里有两种排列:① 数据集**存储**的原始 43 维列(顺序由 `modality.json` 定义,left_hand 排在 right_arm 前、且不含 projected_gravity);② 推理时 sonic 客户端**发送**的(按 modality config 的 `modality_keys` 顺序,right_arm 在前、且多了 gravity)。两者都是官方 SONIC,只是「存盘格式」和「喂模型格式」恰好把 left_hand/right_arm 排反了。
**做法(已固化进代码)**:`SonicInputs` 训练时把 43 维列按数据集**自己的 `modality.json`** 自动切成各组、再按 `modality_keys` 顺序拼成 46 维;推理时桥接也按同一顺序发。**位置从 modality.json 自动读取、不手敲,任何 SONIC 数据集都 by construction 对齐**——换数据集你不用碰这块,也不必担心错位。
**思想**:模型对「第 i 维是什么」是死记的,所以两端必须逐维对齐;我们把这个对齐交给数据集自己的 `modality.json` 来保证,而不是靠人记。


### 4. 为什么是双进程,不是塞一个进程
openpi(JAX + 它自己 patch 过的 transformers)和 GR00T(PyTorch + 另一套 transformers)的依赖会打架。
**做法**:各自待在自己的 venv 里,中间只用 websocket 传 numpy 数组。GR00T 侧的桥接只依赖纯 python 的 `openpi-client`(numpy/msgpack/websockets),零冲突。
**好处**:旧的 sonic 客户端连的还是 GR00T 的 5550 端口,协议不变,一行不用改。

---

## 六、触觉模态(HTD touch-dreaming)

在上面的 `pi0.5-SONIC` 上加触觉,核心一句话:**触觉既当 action head 的额外输入(context),又用一个自监督辅助任务逼模型"看懂"触觉**。一个 `use_tactile` 开关,关掉 = 和原模型字节级一致。

**数据**:`observation.tactile_raw` 是 uint8[256](112 个真实触点 + 保留位),极稀疏(85% 为 0)、时间自相关 0.997。只 `/255`,**不过数据集归一化**,也**不做原始帧堆叠**(没用)。

**两条路(训练时)**:
```
当前帧触觉 ──编码器──► 8 个 token ──注入到 action expert(action token 之前)──► 给动作决策当上下文
未来4帧触觉 ──EMA teacher 编码──► 目标 latent z*  ┐
action expert 在那 8 个触觉位的输出 ──dream head──► 预测 ẑ  ┘──► L_dream = 1-cos(ẑ,z*) + 幅度项
总损失 = 动作 flow-matching 损失 + λ·L_dream
```

**三个关键设计**:
1. **注入点 = action expert(suffix)**:pi0.5 的 action expert 宽 1024,正好等于规格要求的 token 宽。8 个触觉 token 拼在 action token 前、设成 action 能注意到的 context,动作解码切片(最后 H 个)不变。
2. **预测"未来 latent"而非"当前原始触觉"**:HTD 的核心——逼表征编码触觉的时间动态,而不是死记当前读数。target 由一个 **EMA teacher**(student 编码器的滑动平均副本,冻结不回传)产生,避免目标抖动/坍缩。
3. **零部署开销**:teacher 和 dream head **只在训练用**;推理只跑编码器吃当前帧。`use_tactile=False` 时这些模块根本不建。

**为什么不换框架**:EMA teacher 在 JAX 里是把 teacher 作为冻结子模块、每个训练步手动做 `teacher ← decay·teacher + (1-decay)·student` 的拷贝(不进优化器),从而复用已验证的 JAX 训练链,不动 serve/桥接。

> 触觉 layout(112 个有效点的索引、分区)直接复用 GR00T 的 `tactile_layout.py`,同套件同数据。三种编码器(MLP / CNN / CoordConv)都已实现,`--model.tactile-encoder-type` 切换。
>
> **CNN 从 PyTorch 移到 JAX 的三个坑(已验证)**:① Flax `nnx.Conv` 是 NHWC(`[B,H,W,C]`),PyTorch 是 NCHW → region reshape 成 `[B,r,c,1]`;② JAX 没有 `AdaptiveAvgPool2d`,用平均矩阵自实现(已逐元素对齐 PyTorch,含 5×8→2×2 这种不整除分区);③ CoordConv 坐标通道拼在最后(通道)轴、乘 `0.1`。
