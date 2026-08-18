import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.models import tactile as _tactile
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")


def pool_future_vision_embeddings(view_patch_embeddings):
    """Pool [B, T, patches, D] embeddings over patches and camera views."""
    if not view_patch_embeddings:
        raise ValueError("vision-JEPA requires at least one future camera view")
    pooled_views = [embedding.mean(axis=2) for embedding in view_patch_embeddings]
    return jnp.stack(pooled_views, axis=2).mean(axis=2)


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


class Pi0(_model.BaseModel):
    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        if config.pi05:
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        # ---- Tactile fusion and JEPA auxiliary targets ----
        # Modules are only built for their active mode, so notac remains the baseline graph.
        self.use_tactile = config.use_tactile
        self.use_tactile_dream = config.use_tactile_dream
        self.dream_state = config.dream_state
        self.dream_vision = config.dream_vision
        self.use_tactile_temporal = config.use_tactile_temporal
        self.tactile_history_length = config.tactile_history_length if config.use_tactile_temporal else 1
        self.use_delta_targets = config.use_delta_targets
        if config.use_tactile:
            embed = action_expert_config.width
            self.tactile_encoder = _tactile.TactileEncoder(
                encoder_type=config.tactile_encoder_type, embed=embed, rngs=rngs
            )
            if self.use_tactile_temporal:
                self.tactile_temporal_encoder = _tactile.TactileTemporalEncoder(
                    embed, self.tactile_history_length, rngs=rngs
                )
        if config.use_tactile_dream:
            self.dream_horizon = config.dream_horizon
            self.tactile_dream_beta = config.tactile_dream_beta
            # Frozen EMA teacher; starts identical to the student, updated each step in train.py.
            self.tactile_teacher = _tactile.TactileEncoder(
                encoder_type=config.tactile_encoder_type, embed=embed, rngs=rngs
            )
            nnx.update(self.tactile_teacher, nnx.state(self.tactile_encoder))
            self.dream_head = _tactile.DreamHead(embed, embed, tau=config.dream_horizon, rngs=rngs)
        if config.dream_state:
            self.state_encoder = _tactile.StateEncoder(config.action_dim, embed, rngs=rngs)
            self.state_teacher = _tactile.StateEncoder(config.action_dim, embed, rngs=rngs)
            nnx.update(self.state_teacher, nnx.state(self.state_encoder))
            self.state_dream_head = _tactile.DreamHead(embed, embed, tau=config.dream_horizon, rngs=rngs)
        if config.dream_vision:
            self.vision_horizon = config.vision_horizon
            self.vision_dream_head = _tactile.DreamHead(
                embed, paligemma_config.width, tau=config.vision_horizon, rngs=rngs
            )

        # This attribute gets automatically set by model.train() and model.eval().
        self.deterministic = True

    def sync_jepa_teachers(self):
        """Resync EMA teachers after loading a tactile-free base checkpoint."""
        if self.use_tactile_dream:
            nnx.update(self.tactile_teacher, nnx.state(self.tactile_encoder))
        if self.dream_state:
            nnx.update(self.state_teacher, nnx.state(self.state_encoder))

    @staticmethod
    def _current_state(state):
        return state[:, 0] if state.ndim == 3 else state

    def _tactile_features(self, tactile, batch_size: int):
        if tactile is None:
            tactile = jnp.zeros((batch_size, 1, _tactile.RAW_DIM), dtype=jnp.uint8)
        elif tactile.ndim == 2:
            tactile = tactile[:, None]
        history = tactile[:, : self.tactile_history_length]
        if history.shape[1] < self.tactile_history_length:
            padding = jnp.repeat(history[:, :1], self.tactile_history_length - history.shape[1], axis=1)
            history = jnp.concatenate([padding, history], axis=1)
        if not self.use_tactile_temporal:
            return self.tactile_encoder(history[:, -1])
        batch, steps, width = history.shape
        tokens = self.tactile_encoder(history.reshape(batch * steps, width))
        return self.tactile_temporal_encoder(
            tokens.reshape(batch, steps, self.tactile_encoder.n, self.tactile_encoder.embed)
        )

    def _validate_jepa_targets(self, observation: _model.Observation):
        if self.use_tactile and observation.tactile is None:
            raise ValueError("tactile fusion training requires a current tactile frame")
        if not self.use_tactile_dream:
            return
        expected_tactile = self.tactile_history_length + self.dream_horizon
        if observation.tactile.ndim != 3 or observation.tactile.shape[1] != expected_tactile:
            raise ValueError(
                f"tactile dream requires [B, {expected_tactile}, {_tactile.RAW_DIM}], got {observation.tactile.shape}"
            )
        expected_state = self.dream_horizon + 1
        if self.dream_state and (observation.state.ndim != 3 or observation.state.shape[1] != expected_state):
            raise ValueError(f"state-JEPA requires [B, {expected_state}, D], got {observation.state.shape}")
        if self.dream_vision:
            if not observation.future_images:
                raise ValueError("vision-JEPA requires future_images during training")
            for name, image in observation.future_images.items():
                if image.ndim != 5 or image.shape[1] != self.vision_horizon:
                    raise ValueError(
                        f"future image {name!r} must be [B, {self.vision_horizon}, H, W, C], got {image.shape}"
                    )

    def _future_vision_targets(self, future_images):
        view_embeddings = []
        for image in future_images.values():
            batch, horizon = image.shape[:2]
            flat = image.reshape(batch * horizon, *image.shape[2:])
            patch_embeddings, _ = self.PaliGemma.img(flat, train=False)
            view_embeddings.append(patch_embeddings.reshape(batch, horizon, *patch_embeddings.shape[1:]))
        return pool_future_vision_embeddings(view_embeddings)

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        # embed images
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)

            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # image tokens attend to each other
            ar_mask += [False] * image_tokens.shape[1]

        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            # full attention between image and language inputs
            ar_mask += [False] * tokenized_inputs.shape[1]
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    @at.typecheck
    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | None,
    ]:
        input_mask = []
        ar_mask = []
        tokens = []
        if not self.pi05:
            # add a single state token
            current_state = self._current_state(obs.state)
            state_token = self.state_proj(current_state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((current_state.shape[0], 1), dtype=jnp.bool_))
            # image/language inputs do not attend to state or actions
            ar_mask += [True]

        # Without this online token, pi0.5's state encoder would receive no action gradient
        # and its EMA teacher would only track a randomly initialized student.
        if self.dream_state:
            state_token = self.state_encoder(self._current_state(obs.state))[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones(state_token.shape[:2], dtype=jnp.bool_))
            ar_mask += [True]

        action_tokens = self.action_in_proj(noisy_actions)
        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        if self.pi05:
            # time MLP (for adaRMS)
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            action_expert_tokens = action_tokens
            adarms_cond = time_emb
        else:
            # mix timestep + action information using an MLP (no adaRMS)
            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None

        # Inject tactile context tokens BEFORE the action tokens. They form their own block
        # (ar_mask True at the boundary) so the prefix cannot attend to them, but the action
        # tokens (a later block) can attend to them as context. The action-decode slice stays
        # the last `action_horizon` tokens, so action decoding is unchanged.
        if self.use_tactile:
            tactile_tokens = self._tactile_features(
                obs.tactile, action_expert_tokens.shape[0]
            ).astype(action_expert_tokens.dtype)
            n_tac = tactile_tokens.shape[1]
            tokens.append(tactile_tokens)
            input_mask.append(jnp.ones((tactile_tokens.shape[0], n_tac), dtype=jnp.bool_))
            ar_mask += [True] + ([False] * (n_tac - 1))

        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        # image/language/state inputs do not attend to action tokens
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ):
        # Dream mode returns structured auxiliary losses so each JEPA branch is logged separately.
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)
        self._validate_jepa_targets(observation)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # one big forward pass of prefix + suffix at once
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        chunked_loss = jnp.mean(jnp.square(v_t - u_t), axis=-1)

        if not self.use_tactile_dream:
            return chunked_loss

        # Derive the tactile slice from the active token layout: baseline state (pi0 only),
        # online state-JEPA token (optional), tactile slots, then action tokens.
        n_tac = self.tactile_encoder.n
        tac_start = int(not self.pi05) + int(self.dream_state)
        trunk = suffix_out[:, tac_start : tac_start + n_tac].mean(axis=1)  # [b, width]
        z_hat = self.dream_head(trunk)  # [b, tau, embed]
        current_index = self.tactile_history_length - 1
        future = observation.tactile[:, current_index + 1 : current_index + 1 + self.dream_horizon]
        z_star = jax.lax.stop_gradient(self.tactile_teacher.encode_pooled(future))  # [b, tau, embed]
        current_tactile = jax.lax.stop_gradient(
            self.tactile_teacher.encode_pooled(observation.tactile[:, current_index])
        )
        z_star = _tactile.latent_prediction_target(
            z_star, current_tactile, use_delta=self.use_delta_targets
        )
        aux = {
            "tactile_loss": _tactile.dream_loss(z_hat, z_star, beta=self.tactile_dream_beta),
            "state_jepa_loss": jnp.asarray(0.0, dtype=jnp.float32),
            "vision_jepa_loss": jnp.asarray(0.0, dtype=jnp.float32),
        }

        if self.dream_state:
            state_hat = self.state_dream_head(trunk)
            future_state = observation.state[:, 1 : 1 + self.dream_horizon]
            state_target = jax.lax.stop_gradient(self.state_teacher(future_state))
            current_state = jax.lax.stop_gradient(self.state_teacher(observation.state[:, 0]))
            state_target = _tactile.latent_prediction_target(
                state_target, current_state, use_delta=self.use_delta_targets
            )
            aux["state_jepa_loss"] = _tactile.dream_loss(state_hat, state_target, beta=self.tactile_dream_beta)

        if self.dream_vision:
            vision_hat = self.vision_dream_head(trunk)
            vision_target = jax.lax.stop_gradient(self._future_vision_targets(observation.future_images))
            current_vision = jax.lax.stop_gradient(
                pool_future_vision_embeddings(
                    [
                        self.PaliGemma.img(observation.images[name], train=False)[0][:, None]
                        for name in observation.future_images
                    ]
                )[:, 0]
            )
            vision_target = _tactile.latent_prediction_target(
                vision_target, current_vision, use_delta=self.use_delta_targets
            )
            aux["vision_jepa_loss"] = _tactile.dream_loss(vision_hat, vision_target, beta=self.tactile_dream_beta)

        return chunked_loss, aux

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        observation = _model.preprocess_observation(None, observation, train=False)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0
