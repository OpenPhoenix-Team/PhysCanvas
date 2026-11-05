from __future__ import annotations

from flax import nnx
import jax
import jax.numpy as jnp


class TactileEncoder(nnx.Module):
    """Shared CNN+MLP encoder that emits one token per tactile sensor."""

    def __init__(
        self,
        *,
        num_sensors: int = 4,
        points_per_sensor: int = 700,
        input_channels: int = 3,
        grid_height: int = 35,
        grid_width: int = 20,
        hidden_dim: int = 1024,
        output_dim: int = 2048,
        gate_init: float = 0.01,
        output_dtype: str = "bfloat16",
        rngs: nnx.Rngs,
    ):
        if grid_height * grid_width != points_per_sensor:
            raise ValueError(f"tactile grid {grid_height}x{grid_width} does not match {points_per_sensor} points")
        self.num_sensors = num_sensors
        self.points_per_sensor = points_per_sensor
        self.input_channels = input_channels
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.output_dim = output_dim
        self.output_dtype = jnp.dtype(output_dtype)

        self.conv1 = nnx.Conv(input_channels, 64, kernel_size=(3, 3), strides=(2, 2), padding="SAME", rngs=rngs)
        self.norm1 = nnx.LayerNorm(64, rngs=rngs)
        self.conv2 = nnx.Conv(64, 128, kernel_size=(3, 3), strides=(2, 2), padding="SAME", rngs=rngs)
        self.norm2 = nnx.LayerNorm(128, rngs=rngs)
        self.conv3 = nnx.Conv(128, 256, kernel_size=(3, 3), strides=(2, 2), padding="SAME", rngs=rngs)
        self.norm3 = nnx.LayerNorm(256, rngs=rngs)

        # SAME padding with three stride-2 convolutions maps 35x20 to 5x3.
        self.spatial_projection = nnx.Linear(5 * 3 * 256, hidden_dim, rngs=rngs)
        self.hidden_norm = nnx.LayerNorm(hidden_dim, rngs=rngs)
        self.output_projection = nnx.Linear(hidden_dim, output_dim, rngs=rngs)
        self.output_norm = nnx.LayerNorm(output_dim, rngs=rngs)

        self.sensor_embedding = nnx.Embed(num_sensors, output_dim, rngs=rngs)
        self.modality_embedding = nnx.Param(jnp.zeros((1, 1, output_dim), dtype=jnp.float32))
        self.gate = nnx.Param(jnp.asarray(gate_init, dtype=jnp.float32))

    def __call__(self, tactile: jax.Array) -> jax.Array:
        if tactile.ndim != 4:
            raise ValueError(f"expected tactile [B,S,P,C], got shape {tactile.shape}")
        expected = (self.num_sensors, self.points_per_sensor, self.input_channels)
        if tactile.shape[1:] != expected:
            raise ValueError(f"expected tactile trailing shape {expected}, got {tactile.shape[1:]}")

        batch_size = tactile.shape[0]
        x = jnp.asarray(tactile, dtype=jnp.float32)
        x = x.reshape(batch_size * self.num_sensors, self.grid_height, self.grid_width, self.input_channels)
        x = jax.nn.gelu(self.norm1(self.conv1(x)))
        x = jax.nn.gelu(self.norm2(self.conv2(x)))
        x = jax.nn.gelu(self.norm3(self.conv3(x)))
        x = x.reshape(batch_size * self.num_sensors, -1)
        x = jax.nn.gelu(self.hidden_norm(self.spatial_projection(x)))
        x = self.output_norm(self.output_projection(x))
        x = x.reshape(batch_size, self.num_sensors, self.output_dim)

        sensor_ids = jnp.arange(self.num_sensors, dtype=jnp.int32)
        sensor_embedding = self.sensor_embedding(sensor_ids)[None, :, :]
        x = self.gate[...] * (x + sensor_embedding + self.modality_embedding[...])
        return x.astype(self.output_dtype)
