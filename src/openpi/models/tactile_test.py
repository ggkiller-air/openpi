import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import tactile


def test_sonic_layout_is_a_valid_partition():
    assert len(tactile.VALID_IDX) == tactile.NUM_VALID == sum(tactile.REGION_SIZES)
    assert len(set(tactile.VALID_IDX)) == tactile.NUM_VALID
    assert min(tactile.VALID_IDX) >= 0
    assert max(tactile.VALID_IDX) < tactile.RAW_DIM
    assert tuple(rows * cols for rows, cols in tactile.REGION_GRIDS) == tactile.REGION_SIZES


def test_tactile_normalization_happens_once():
    encoder = tactile.TactileEncoder(embed=16, hidden=8, n=2, rngs=nnx.Rngs(0))
    raw = jnp.stack(
        [jnp.zeros((tactile.RAW_DIM,)), jnp.full((tactile.RAW_DIM,), 255.0)],
        axis=0,
    )
    selected = encoder.select_and_normalize(raw)
    np.testing.assert_array_equal(np.asarray(selected[0]), 0.0)
    np.testing.assert_array_equal(np.asarray(selected[1]), 1.0)


@pytest.mark.parametrize("encoder_type", ["mlp", "cnn", "coord"])
def test_tactile_encoder_variants_shape_and_input_gradient(encoder_type):
    encoder = tactile.TactileEncoder(encoder_type=encoder_type, embed=16, hidden=8, n=3, rngs=nnx.Rngs(0))
    raw = jnp.full((2, tactile.RAW_DIM), 127.0)

    output = encoder(raw)
    gradient = jax.grad(lambda value: encoder(value).sum())(raw)

    assert output.shape == (2, 3, 16)
    assert gradient.shape == raw.shape
    assert bool(jnp.all(jnp.isfinite(output)))
    assert bool(jnp.all(jnp.isfinite(gradient)))


def test_state_encoder_preserves_time_dimension():
    encoder = tactile.StateEncoder(12, 16, hidden=8, rngs=nnx.Rngs(0))
    state = jnp.ones((2, 4, 12), dtype=jnp.float32)
    assert encoder(state).shape == (2, 4, 16)


def test_dream_loss_is_finite_and_zero_for_identical_nonzero_latents():
    latent = jnp.ones((2, 4, 16), dtype=jnp.float32)
    loss = tactile.dream_loss(latent, latent)
    assert bool(jnp.isfinite(loss))
    np.testing.assert_allclose(np.asarray(loss), 0.0, atol=1e-6)
