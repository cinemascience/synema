import flax.linen as nn
import jax.numpy as jnp
import jax.random

from encoders.frequency import PositionalEncodingNeRF
from encoders.hashgrid import HashGridEncoder
from models.siren import Sine


class CinemaScalarImage(nn.Module):
    num_hidden_features = 128
    # position_encoder: HashGridEncoder = HashGridEncoder(num_levels=8, feature_dims=4, table_size=2 ** 19)
    position_encoder: PositionalEncodingNeRF = PositionalEncodingNeRF(num_frequencies=10)

    def init_last(self, key, shape, dtype):
        v = jnp.sqrt(6. / shape[0]) / 30.0
        return jax.random.uniform(key=key, shape=shape, dtype=dtype,
                                  minval=-v, maxval=v)

    @nn.compact
    def __call__(self, input_points, input_views=None):
        # Simple MLP with SIREN
        # encoded_points = self.position_encoder(input_points)
        # x = Sine(hidden_features=self.num_hidden_features, is_first=True)(encoded_points)
        # for i in range(4):
        #     x = Sine(hidden_features=self.num_hidden_features, is_first=False)(x)
        #     if i == 3:
        #         x = jnp.concatenate([x, encoded_points], axis=-1)
        # x = nn.Dense(features=2, kernel_init=self.init_last)(x)
        # density, scalar = jnp.split(x, [1], axis=-1)
        # density = nn.relu(density)
        # return scalar, jnp.squeeze(density)

        # Simple MLP
        # encoded_points = self.position_encoder(input_points)
        # x = encoded_points
        # for i in range(6):
        #     x = nn.Dense(features=self.num_hidden_features)(x)
        #     x = nn.relu(x)
        #     if i == 4:
        #         x = jnp.concatenate([x, encoded_points], axis=-1)
        # x = nn.Dense(features=2)(x)
        # density, scalar = jnp.split(x, [1], axis=-1)
        # density = nn.relu(density)
        # return scalar, jnp.squeeze(density)

        encoded_points = self.position_encoder(input_points)
        x = Sine(hidden_features=self.num_hidden_features, is_first=True)(encoded_points)
        x = Sine(hidden_features=self.num_hidden_features, is_first=False)(x)
        x = Sine(hidden_features=self.num_hidden_features, is_first=False)(x)
        x = Sine(hidden_features=self.num_hidden_features, is_first=False)(x)
        x = nn.Dense(features=16, kernel_init=self.init_last)(x)
        density, scalar = jnp.split(x, [1], axis=-1)
        density = nn.relu(density)

        scalar = jnp.concatenate([scalar, encoded_points], axis=-1)
        scalar = Sine(hidden_features=self.num_hidden_features, is_first=True)(scalar)
        scalar = Sine(hidden_features=self.num_hidden_features, is_first=False)(scalar)
        scalar = nn.Dense(features=1, kernel_init=self.init_last)(scalar)

        return scalar, jnp.squeeze(density)

        # encoded_points = self.position_encoder(input_points)
        # # density MLP, two layers of ReLU
        # x = nn.Dense(features=self.num_hidden_features)(encoded_points)
        # x = nn.relu(x)
        # x = nn.Dense(features=self.num_hidden_features)(x)
        # x = nn.relu(x)
        # x = nn.Dense(features=self.num_hidden_features)(x)
        # x = nn.relu(x)
        # x = nn.Dense(features=16)(x)
        #
        # density, scalar = jnp.split(x, [1], axis=-1)
        # density = nn.relu(density)
        #
        # # scalar MLP, two layers of ReLU
        # scalar = jnp.concatenate([scalar, encoded_points], axis=-1)
        # scalar = nn.Dense(features=self.num_hidden_features)(scalar)
        # scalar = nn.relu(scalar)
        # scalar = nn.Dense(features=self.num_hidden_features)(scalar)
        # scalar = nn.relu(scalar)
        # scalar = nn.Dense(features=1)(scalar)
        #
        # return scalar, jnp.squeeze(density)
