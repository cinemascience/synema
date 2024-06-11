import flax.linen as nn
import jax.numpy as jnp

from encoders.frequency import PositionalEncodingNeRF


class CinemaScalarImage(nn.Module):
    num_hidden_features = 64
    # position_encoder: HashGridEncoder = HashGridEncoder(num_levels=8, feature_dims=4, max_resolution=1024)
    position_encoder: PositionalEncodingNeRF = PositionalEncodingNeRF()

    @nn.compact
    def __call__(self, input_points, input_views=None):
        encoded_points = self.position_encoder(input_points)

        # density MLP, two layers of ReLU
        x = nn.Dense(features=self.num_hidden_features)(encoded_points)
        x = nn.relu(x)
        x = nn.Dense(features=self.num_hidden_features)(x)
        x = nn.relu(x)
        x = nn.Dense(features=16)(x)

        density, scalar = jnp.split(x, [1], axis=-1)
        # TODO: use exponential activation?
        density = nn.relu(density)

        # scalar MLP, two layers of ReLU
        scalar = nn.Dense(features=self.num_hidden_features)(scalar)
        scalar = nn.relu(scalar)
        scalar = nn.Dense(features=self.num_hidden_features)(scalar)
        scalar = nn.relu(scalar)
        scalar = nn.Dense(features=1)(scalar)

        return scalar, jnp.squeeze(density)
