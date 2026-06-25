"""Tactile representation module for π0.5 (HTD touch-dreaming), Flax nnx port.

Ports `/data/zihao/tactile_representation_spec.md` (validated on Isaac-GR00T) onto pi0.5.
The encoder turns a raw tactile packet into N conditioning tokens injected into the action
expert; a training-only dream head predicts the *future* tactile latent from the trunk.

Three per-region encoders are provided via `encoder_type`: "mlp", "cnn", "coord" (CoordConv).
All share the same I/O ([B,256] -> [B,N,embed]) and the slot aggregator / dream path.

Sensor layout (Unitree G1 SONIC skin suit) is copied verbatim from
Isaac-GR00T/gr00t/data/tactile_layout.py (same suit, same carry-bucket data).

Widths for pi0.5: D_tok == D_trunk == action-expert width == 1024 (gemma_300m).

JAX/PyTorch CNN porting notes (verified): Flax `nnx.Conv` is NHWC ([B,H,W,C]) vs PyTorch's
NCHW, so regions are reshaped to [B,r,c,1]; JAX has no AdaptiveAvgPool2d so `_adaptive_avg_pool2d`
reimplements it with averaging matrices (matches torch elementwise, incl. non-divisible grids);
coord channels are concatenated on the last (channel) axis and scaled by 0.1 (spec gotcha #3).
"""

import jax
import jax.numpy as jnp
import flax.nnx as nnx
import numpy as np

import openpi.shared.array_typing as at

# ---- Sensor layout (SONIC), region-ordered, 0-based. From GR00T tactile_layout.py. ----
# Region order: front_chest(48), back(40), left_arm(8), left_shoulder(4), right_arm(8), right_shoulder(4).
VALID_IDX: tuple[int, ...] = (
    194, 210, 226, 242, 2, 18, 34, 50, 195, 211, 227, 243, 3, 19, 35, 51, 196, 212, 228, 244,
    4, 20, 36, 52, 197, 213, 229, 245, 5, 21, 37, 53, 198, 214, 230, 246, 6, 22, 38, 54,
    199, 215, 231, 247, 7, 23, 39, 55,  # front_chest 48
    57, 41, 25, 9, 249, 233, 217, 201, 58, 42, 26, 10, 250, 234, 218, 202, 59, 43, 27, 11,
    251, 235, 219, 203, 60, 44, 28, 12, 252, 236, 220, 204, 61, 45, 29, 13, 253, 237, 221, 205,  # back 40
    78, 94, 110, 126, 79, 95, 111, 127,  # left_arm 8
    8, 24, 40, 56,  # left_shoulder 4
    176, 161, 145, 129, 177, 160, 144, 128,  # right_arm 8
    248, 232, 216, 200,  # right_shoulder 4
)
REGION_SIZES: tuple[int, ...] = (48, 40, 8, 4, 8, 4)  # sum == 112
REGION_GRIDS: tuple[tuple[int, int], ...] = ((6, 8), (5, 8), (2, 4), (1, 4), (2, 4), (1, 4))  # cnn/coord only
RAW_DIM = 256
NUM_VALID = 112

# Defaults (spec §5), embed adapted 1536 -> 1024 for pi0.5.
EMBED_DIM = 1024
HIDDEN_DIM = 512
N_TOKENS = 8
NUM_HEADS = 8


class _SmallMLP(nnx.Module):
    def __init__(self, i: int, h: int, o: int, *, rngs: nnx.Rngs):
        self.l1 = nnx.Linear(i, h, rngs=rngs)
        self.l2 = nnx.Linear(h, o, rngs=rngs)

    def __call__(self, x):
        return self.l2(nnx.relu(self.l1(x)))


class PerRegionMLP(nnx.Module):
    """Each region's flat slice [B, size] -> 2-layer MLP -> [B, embed]; stacked to [B, R, embed]."""

    def __init__(self, sizes, hidden: int, embed: int, *, rngs: nnx.Rngs):
        self.sizes = tuple(int(s) for s in sizes)
        # Use nnx.Dict with string keys (NOT a Python list): list submodules get integer keys
        # in the param tree, which break the weight loader's flatten_dict(sep="/").
        self.branches = nnx.Dict({str(i): _SmallMLP(n, hidden, embed, rngs=rngs) for i, n in enumerate(self.sizes)})

    def __call__(self, x):  # x: [B, sum(sizes)] -> [B, R, embed]
        splits = np.cumsum(self.sizes)[:-1].tolist()
        parts = jnp.split(x, splits, axis=-1)
        return jnp.stack([self.branches[str(i)](p) for i, p in enumerate(parts)], axis=1)


def _pool_matrix(in_size: int, out_size: int) -> np.ndarray:
    """Averaging matrix [out, in] matching torch AdaptiveAvgPool1d bin boundaries.

    Output cell i averages input indices [floor(i*I/O), ceil((i+1)*I/O)). Verified to match
    torch.nn.AdaptiveAvgPool2d elementwise (incl. non-divisible sizes like 5->2).
    """
    m = np.zeros((out_size, in_size), np.float32)
    for i in range(out_size):
        s = (i * in_size) // out_size
        e = -(-((i + 1) * in_size) // out_size)  # ceil division
        m[i, s:e] = 1.0 / (e - s)
    return m


def _adaptive_avg_pool2d(x, out_hw):  # x: [B, H, W, C] -> [B, oh, ow, C]
    _, h, w, _ = x.shape
    oh, ow = out_hw
    mh = jnp.asarray(_pool_matrix(h, oh))
    mw = jnp.asarray(_pool_matrix(w, ow))
    return jnp.einsum("ih,bhwc,jw->bijc", mh, x, mw)


def _coord_axis(n: int) -> np.ndarray:
    """Normalized coordinate along an axis of length n: [-1, 1]; a 1-length strip gets coord 0."""
    return np.zeros(1, np.float32) if n == 1 else np.linspace(-1.0, 1.0, n, dtype=np.float32)


class PerRegionCNN(nnx.Module):
    """Each region [B, size] -> reshape to (rows, cols) grid -> single 3x3 conv -> ReLU ->
    adaptive avg pool -> Linear(embed). Optionally prepend 2 normalized coordinate channels
    (CoordConv). Output [B, R, embed]. Exploits per-region spatial layout (conv weight sharing).
    """

    def __init__(
        self,
        grids,
        embed: int,
        *,
        ch: int = 32,
        pool: tuple[int, int] = (2, 2),
        coord: bool = False,
        coord_scale: float = 0.1,  # spec gotcha #3: match coord std to sparse value std
        rngs: nnx.Rngs,
    ):
        self.grids = [(int(r), int(c)) for r, c in grids]
        self.coord = coord
        self.coord_scale = float(coord_scale)
        self.pools = [(min(pool[0], r), min(pool[1], c)) for r, c in self.grids]
        in_ch = 1 + (2 if coord else 0)
        # nnx.Dict with string keys (see PerRegionMLP note): avoids integer param-tree keys.
        self.convs = nnx.Dict(
            {str(i): nnx.Conv(in_ch, ch, kernel_size=(3, 3), padding="SAME", rngs=rngs) for i in range(len(self.grids))}
        )
        self.projs = nnx.Dict(
            {str(i): nnx.Linear(ch * ph * pw, embed, rngs=rngs) for i, (ph, pw) in enumerate(self.pools)}
        )

    def __call__(self, x):  # x: [B, sum(rows*cols)] -> [B, R, embed]
        sizes = [r * c for r, c in self.grids]
        splits = np.cumsum(sizes)[:-1].tolist()
        parts = jnp.split(x, splits, axis=-1)
        out = []
        for i, ((r, c), (ph, pw), p) in enumerate(zip(self.grids, self.pools, parts)):
            conv = self.convs[str(i)]
            proj = self.projs[str(i)]
            b = p.shape[0]
            g = p.reshape(b, r, c, 1)  # NHWC single channel (row-major, matches PyTorch content)
            if self.coord:
                rc = jnp.asarray(_coord_axis(r) * self.coord_scale)  # [r]
                cc = jnp.asarray(_coord_axis(c) * self.coord_scale)  # [c]
                row_ch = jnp.broadcast_to(rc[None, :, None, None], (b, r, c, 1))
                col_ch = jnp.broadcast_to(cc[None, None, :, None], (b, r, c, 1))
                g = jnp.concatenate([g, row_ch, col_ch], axis=-1)  # [B, r, c, 3]
            h = nnx.relu(conv(g))  # [B, r, c, ch]
            h = _adaptive_avg_pool2d(h, (ph, pw))  # [B, ph, pw, ch]
            out.append(proj(h.reshape(b, -1)))  # [B, embed]
        return jnp.stack(out, axis=1)


class SlotAggregator(nnx.Module):
    """N learnable queries cross-attend the R region tokens -> [B, N, embed]."""

    def __init__(self, embed: int, n: int, *, num_heads: int = NUM_HEADS, rngs: nnx.Rngs):
        self.query = nnx.Param(jax.random.normal(rngs.params(), (n, embed)) * 0.02)
        self.attn = nnx.MultiHeadAttention(num_heads=num_heads, in_features=embed, decode=False, rngs=rngs)
        self.norm = nnx.LayerNorm(embed, rngs=rngs)

    def __call__(self, region_tokens):  # [B, R, embed] -> [B, N, embed]
        b = region_tokens.shape[0]
        q = jnp.broadcast_to(self.query.value, (b, *self.query.value.shape))
        a = self.attn(q, region_tokens, region_tokens, deterministic=True)
        return self.norm(a)


class TactileEncoder(nnx.Module):
    """valid-select(112) -> /255 -> per-region encoder -> slot aggregator. Output [B, N, embed]."""

    def __init__(
        self,
        *,
        encoder_type: str = "mlp",
        embed: int = EMBED_DIM,
        hidden: int = HIDDEN_DIM,
        n: int = N_TOKENS,
        rngs: nnx.Rngs,
    ):
        self.embed = embed
        self.n = n
        # NOTE: valid_idx is the module-level VALID_IDX constant used directly in _select_norm.
        # It must NOT be stored on self (nnx would treat an array attribute as state and fail).
        if encoder_type == "mlp":
            self.per_region = PerRegionMLP(REGION_SIZES, hidden, embed, rngs=rngs)
        elif encoder_type == "cnn":
            self.per_region = PerRegionCNN(REGION_GRIDS, embed, coord=False, rngs=rngs)
        elif encoder_type == "coord":
            # CoordConv: 2 normalized coordinate channels scaled by 0.1 (spec gotcha #3).
            self.per_region = PerRegionCNN(REGION_GRIDS, embed, coord=True, coord_scale=0.1, rngs=rngs)
        else:
            raise ValueError(f"unknown tactile encoder_type: {encoder_type!r} (expected mlp|cnn|coord)")
        self.agg = SlotAggregator(embed, n, rngs=rngs)

    def _select_norm(self, raw):  # [..., 256] -> [..., 112] in [0,1]
        sel = jnp.take(raw, jnp.asarray(VALID_IDX, dtype=jnp.int32), axis=-1)
        return sel.astype(self.agg.norm.scale.value.dtype) / 255.0

    def __call__(self, cur):  # [B, 256] -> [B, N, embed]
        return self.agg(self.per_region(self._select_norm(cur)))

    def encode_pooled(self, raw_window):  # [B, T, 256] -> [B, T, embed] (mean over N slots)
        b, t = raw_window.shape[0], raw_window.shape[1]
        tok = self(raw_window.reshape(b * t, raw_window.shape[-1]))  # [B*T, N, embed]
        return tok.mean(axis=1).reshape(b, t, self.embed)


class DreamHead(nnx.Module):
    """Predict the future tactile latent from the trunk feature: [B, d_trunk] -> [B, tau, embed]."""

    def __init__(self, d_trunk: int, embed: int, tau: int, *, hidden: int = HIDDEN_DIM, rngs: nnx.Rngs):
        self.tau = tau
        self.embed = embed
        self.net = _SmallMLP(d_trunk, hidden, tau * embed, rngs=rngs)

    def __call__(self, x):  # [B, d_trunk] -> [B, tau, embed]
        return self.net(x).reshape(x.shape[0], self.tau, self.embed)


def dream_loss(pred, target, beta: float = 1.0):  # [B, tau, embed] each; target detached
    """HTD loss: direction (1 - cos) + beta * smooth_l1 on latent norms. Computed in fp32."""
    p = pred.astype(jnp.float32)
    t = target.astype(jnp.float32)
    direction = 1.0 - _cosine_similarity(p, t, axis=-1)
    magnitude = _smooth_l1(jnp.linalg.norm(p, axis=-1), jnp.linalg.norm(t, axis=-1))
    return jnp.mean(direction + beta * magnitude)


def _cosine_similarity(a, b, axis=-1, eps=1e-8):
    num = jnp.sum(a * b, axis=axis)
    den = jnp.linalg.norm(a, axis=axis) * jnp.linalg.norm(b, axis=axis)
    return num / jnp.maximum(den, eps)


def _smooth_l1(a, b, beta: float = 1.0):
    d = jnp.abs(a - b)
    return jnp.where(d < beta, 0.5 * d * d / beta, d - 0.5 * beta)
