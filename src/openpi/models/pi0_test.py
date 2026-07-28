import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import pi0
import openpi.models.pi0_config as _pi0_config
from openpi.training import config as _training_config


def _get_frozen_state(config: _pi0_config.Pi0Config) -> nnx.State:
    abstract_model = nnx.eval_shape(config.create, jax.random.key(0))

    freeze_filter = config.get_freeze_filter()
    return nnx.state(abstract_model, nnx.All(nnx.Param, freeze_filter)).flat_state()


def test_pi0_full_finetune():
    config = _pi0_config.Pi0Config()
    state = _get_frozen_state(config)
    assert len(state) == 0


def test_pi0_gemma_lora():
    config = _pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora")
    state = _get_frozen_state(config)
    assert len(state) == 9
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)
    assert all("_1" not in p for p in state)


def test_pi0_action_expert_lora():
    config = _pi0_config.Pi0Config(action_expert_variant="gemma_300m_lora")
    state = _get_frozen_state(config)
    # excluding embedder, rest of the params should be same as gemma_lora.
    assert len(state) == 8
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)
    # all frozen params should have _1 in their path since it's the action expert.
    assert all(any("_1" in p for p in path) for path in state)


def test_pi0_all_lora():
    config = _pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora")
    state = _get_frozen_state(config)
    # sum of gemma_lora and action_expert_lora's frozen params.
    assert len(state) == 17
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)


def test_jepa_mode_validation():
    with pytest.raises(ValueError, match="requires use_tactile=True"):
        _pi0_config.Pi0Config(use_tactile_dream=True)
    with pytest.raises(ValueError, match="require use_tactile_dream=True"):
        _pi0_config.Pi0Config(use_tactile=True, dream_state=True)
    with pytest.raises(ValueError, match="require use_tactile_dream=True"):
        _pi0_config.Pi0Config(use_tactile=True, dream_vision=True)


def test_jepa_input_specs_match_current_and_future_contract():
    input_only = _pi0_config.Pi0Config(use_tactile=True)
    input_obs, _ = input_only.inputs_spec(batch_size=2)
    assert input_obs.tactile.shape == (2, 1, 256)
    assert input_obs.state.shape == (2, input_only.action_dim)
    assert input_obs.future_images is None

    dream = _pi0_config.Pi0Config(
        use_tactile=True,
        use_tactile_dream=True,
        dream_state=True,
        dream_vision=True,
        dream_horizon=3,
        vision_horizon=2,
    )
    dream_obs, _ = dream.inputs_spec(batch_size=2)
    assert dream_obs.tactile.shape == (2, 4, 256)
    assert dream_obs.state.shape == (2, 4, dream.action_dim)
    assert all(image.shape == (2, 2, 224, 224, 3) for image in dream_obs.future_images.values())


def test_future_vision_pooling_preserves_time_and_averages_views_and_patches():
    first = jnp.asarray([[[[0.0], [2.0]], [[10.0], [14.0]]]])
    second = jnp.asarray([[[[4.0], [6.0]], [[18.0], [22.0]]]])

    pooled = pi0.pool_future_vision_embeddings([first, second])

    assert pooled.shape == (1, 2, 1)
    np.testing.assert_allclose(np.asarray(pooled), [[[3.0], [16.0]]])


def test_jepa_module_graph_and_frozen_targets():
    input_model = nnx.eval_shape(_pi0_config.Pi0Config(use_tactile=True).create, jax.random.key(0))
    assert hasattr(input_model, "tactile_encoder")
    assert not hasattr(input_model, "tactile_teacher")
    assert not hasattr(input_model, "dream_head")

    model_config = _pi0_config.Pi0Config(
        pi05=True,
        action_dim=78,
        action_horizon=40,
        discrete_state_input=False,
        use_tactile=True,
        use_tactile_dream=True,
        dream_state=True,
        dream_vision=True,
    )
    model = nnx.eval_shape(model_config.create, jax.random.key(0))
    expected_modules = (
        "tactile_encoder",
        "tactile_teacher",
        "dream_head",
        "state_encoder",
        "state_teacher",
        "state_dream_head",
        "vision_dream_head",
    )
    assert all(hasattr(model, name) for name in expected_modules)

    train_config = _training_config.TrainConfig(name="filter_test", model=model_config)
    all_paths = ["/".join(str(part) for part in path) for path in nnx.state(model, nnx.Param).flat_state()]
    trainable_paths = [
        "/".join(str(part) for part in path) for path in nnx.state(model, train_config.trainable_filter).flat_state()
    ]
    assert any("PaliGemma/img" in path for path in all_paths)
    assert not any("PaliGemma/img" in path for path in trainable_paths)
    assert not any("teacher" in path for path in trainable_paths)
