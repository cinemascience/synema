import flax.linen as nn
import jax.numpy as jnp
import jax
from synema.encoders.frequency import PositionalEncodingNeRF
from synema.encoders.hashgrid import HashGridEncoder
from synema.encoders.sh4 import SphericalHarmonic4thEncoder
from synema.models.siren import Siren, Sine


class Cinema(nn.Module):
    @staticmethod
    def normalize_points(input_points):
        # We constructed the columns of the camera pose matrix for Cinema
        # database as normalized vectors, depth values are also in the range
        # of [0, 1], making the points coordinates in the bbox of [-1, 1],
        # re-normalize to [0, 1] for hashgrid encoding.
        input_points = input_points + jnp.ones_like(input_points)
        input_points = input_points / 2.
        return input_points


class CinemaRGBAImage(Cinema):
    num_hidden_features: int = 64
    omega_0: float = 30.
    position_encoder: nn.Module = PositionalEncodingNeRF(num_frequencies=10)
    view_encoder: nn.Module = SphericalHarmonic4thEncoder()

    def init_last(self, key, shape, dtype):
        v = jnp.sqrt(6. / shape[0]) / self.omega_0
        return jax.random.uniform(key=key, shape=shape, dtype=dtype,
                                  minval=-v, maxval=v)

    @nn.compact
    def __call__(self, input_points, input_views):
        input_points = self.normalize_points(input_points)
        encoded_points = self.position_encoder(input_points)

        # Note: it is empirically found that the number of density layers
        # needs to be larger than the number of color layers.
        # Otherwise, there are floats in the reconstructed depth image.
        # density MLP, 2 layers of SIREN
        x = Sine(hidden_features=self.num_hidden_features, is_first=True)(encoded_points)
        x = Sine(hidden_features=self.num_hidden_features)(x)
        x = Sine(hidden_features=self.num_hidden_features)(x)
        x = nn.Dense(features=16, kernel_init=self.init_last)(x)

        # we use relu instead of exp since we do want density == 0 for empty space.
        densities = nn.relu(x[..., 0])

        # color MLP, 1 layers of SIREN
        encoded_dir = self.view_encoder(input_views)
        x = jnp.concatenate([x[..., 1:], encoded_dir], axis=-1)

        x = Sine(hidden_features=self.num_hidden_features)(x)
        x = Sine(hidden_features=self.num_hidden_features)(x)
        x = nn.Dense(features=3, kernel_init=self.init_last)(x)

        colors = nn.sigmoid(x)
        return colors, densities


class CinemaScalarImage(Cinema):
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
        input_points = self.normalize_points(input_points)
        encoded_points = self.position_encoder(input_points)

        x = Siren(hidden_features=self.num_hidden_features, hidden_layers=4, out_features=16)(encoded_points)
        density, scalar = jnp.split(x, [1], axis=-1)
        density = nn.relu(density)

        scalar = jnp.concatenate([scalar, encoded_points], axis=-1)
        scalar = Siren(hidden_features=self.num_hidden_features, hidden_layers=2)(scalar)
        return scalar, jnp.squeeze(density)
