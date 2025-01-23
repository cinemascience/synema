import flax.linen as nn
import jax.numpy as jnp

from synema.encoders.hashgrid import HashGridEncoder
from synema.models.siren import Siren


class CinemaScalarImage(nn.Module):
    num_hidden_features = 64
    # When using SIREN, the hash_init_scale needs to be larger than described
    # in the InstantNGP paper to prevent vanishing gradient.
    # The goal is to make the encoded features to stretch across [-1, 1] as
    # possible but not to "overflow" the [-1, 1] range when multiplied by the
    # weights in the first layer in SIREN.
    position_encoder: HashGridEncoder = HashGridEncoder(num_levels=8,
                                                        table_size=2 ** 19,
                                                        feature_dims=4,
                                                        max_resolution=2 ** 12,
                                                        hash_init_scale=1.e-1)

    @nn.compact
    def __call__(self, input_points, input_views=None):
        # We constructed the columns of the camera pose matrix for Cinema
        # database normalized vectors, depth values are also in the range
        # of [0, 1], making the points coordinates in the bbox of [-1, 1],
        # re-normalize to [0, 1] for hashgrid encoding.
        input_points = input_points + jnp.ones_like(input_points)
        input_points = input_points / 2.

        encoded_points = self.position_encoder(input_points)
        x = Siren(hidden_features=self.num_hidden_features, hidden_layers=4, out_features=16)(encoded_points)
        density, scalar = jnp.split(x, [1], axis=-1)
        density = nn.relu(density)

        scalar = jnp.concatenate([scalar, encoded_points], axis=-1)
        scalar = Siren(hidden_features=self.num_hidden_features, hidden_layers=2)(scalar)
        return scalar, jnp.squeeze(density)
