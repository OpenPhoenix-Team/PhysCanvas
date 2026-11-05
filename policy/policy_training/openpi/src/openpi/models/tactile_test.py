import flax.nnx as nnx
import jax
import jax.numpy as jnp
import pytest

from openpi.models import tactile as _tactile


def test_tactile_encoder_emits_one_token_per_sensor():
    encoder = _tactile.TactileEncoder(rngs=nnx.Rngs(jax.random.key(0)))
    tactile = jax.random.normal(jax.random.key(1), (2, 4, 700, 3))

    tokens = encoder(tactile)

    assert tokens.shape == (2, 4, 2048)
    assert tokens.dtype == jnp.bfloat16
    assert bool(jnp.all(jnp.isfinite(tokens)))
    # Shared CNN/MLP tokens are disambiguated by learned sensor IDs.
    assert not bool(jnp.all(tokens[:, 0] == tokens[:, 1]))


def test_tactile_encoder_rejects_wrong_layout():
    encoder = _tactile.TactileEncoder(rngs=nnx.Rngs(jax.random.key(0)))
    with pytest.raises(ValueError, match="expected tactile trailing shape"):
        encoder(jnp.zeros((1, 4, 699, 3), dtype=jnp.float32))
