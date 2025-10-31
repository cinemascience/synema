from dataclasses import dataclass

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from synema.encoders.frequency import PositionalEncodingNeRF
from synema.encoders.hashgrid import HashGridEncoder
from synema.encoders.sh4 import SphericalHarmonic4thEncoder
from synema.models.siren import Sine, SirenResidualBlock
from synema.models.wire import Wire


def raw2output(x: ArrayLike) -> (ArrayLike, ArrayLike):
    # Turn the raw output of the NN as color and density, also making them
    # within [0, 1] range as described in the NeRF paper.
    colors = nn.sigmoid(x[..., :3])
    density = nn.relu(x[..., 3])

    return colors, density


@dataclass
class NeRF(nn.Module):
    aabb: jnp.array = None
    # normalize input coordinates to the range of [0, 1]^3 to work with SIREN
    # and HashGrid.
    def normalize_points(self, input_points):
        input_points = input_points - self.aabb[0]
        input_points = input_points / (self.aabb[1] - self.aabb[0])
        return input_points


class NeRFModel(NeRF):
    r"""NeRF model as described in the original paper https://arxiv.org/abs/2003.08934"""

    num_hidden_layers: int = 8
    num_hidden_features: int = 256
    use_viewdirs: bool = True
    position_encoder: PositionalEncodingNeRF = PositionalEncodingNeRF(num_frequencies=10)
    view_encoder: PositionalEncodingNeRF = PositionalEncodingNeRF(num_frequencies=4)

    @nn.compact
    def __call__(self, input_points, input_views=None):
        input_points = self.normalize_points(input_points)
        input_points = self.position_encoder(input_points)
        x = input_points
        for i in range(self.num_hidden_layers):
            x = nn.Dense(features=self.num_hidden_features)(x)
            x = nn.relu(x)
            if i == 4:
                x = jnp.concatenate([x, input_points], axis=-1)

        if self.use_viewdirs and input_views is not None:
            input_views = self.view_encoder(input_views)

            sigma_out = nn.Dense(features=1)(x)
            bottleneck = nn.Dense(features=self.num_hidden_features)(x)
            input_views = jnp.concatenate([bottleneck, input_views], axis=-1)

            x = nn.Dense(features=self.num_hidden_features // 2)(input_views)
            x = nn.relu(x)
            x = nn.Dense(features=3)(x)
            x = jnp.concatenate([x, sigma_out], axis=-1)
        else:
            x = nn.Dense(features=4)(x)

        return raw2output(x)


class TinyNeRFModel(NeRF):
    r"""Simplified NeRF Model from https://github.com/bmild/nerf/blob/master/tiny_nerf.ipynb that
    does not use view directions as an input"""

    num_hidden_layers: int = 8
    num_hidden_features: int = 128
    apply_positional_encoding: bool = True

    @nn.compact
    def __call__(self, input_points, *args, **kwargs):
        input_points = self.normalize_points(input_points)
        input_points = PositionalEncodingNeRF()(input_points) if self.apply_positional_encoding else input_points
        x = input_points
        for i in range(self.num_hidden_layers):
            x = nn.Dense(features=self.num_hidden_features)(x)
            x = nn.relu(x)
            if i == 4:
                x = jnp.concatenate([x, input_points], axis=-1)
        x = nn.Dense(features=4)(x)
        return raw2output(x)


class VeryTinyNeRFModel(NeRF):
    r"""Very tiny NeRF model from nerf-pytorch, https://github.com/krrish94/nerf-pytorch"""
    num_hidden_features: int = 128
    num_encoding_functions: int = 6
    position_encoder: PositionalEncodingNeRF = PositionalEncodingNeRF(num_encoding_functions)

    @nn.compact
    def __call__(self, input_points, *args, **kwargs):
        input_points = self.normalize_points(input_points)
        x = self.position_encoder(input_points)
        for i in range(3):
            x = nn.Dense(features=self.num_hidden_features)(x)
            x = nn.relu(x)
        x = nn.Dense(features=4)(x)
        return raw2output(x)


class InstantNGP(NeRF):
    r"""
    Model from the InstantNGP paper https://arxiv.org/abs/2201.05989 with modification as
     implemented by nerfstudio.
    """
    num_hidden_features: int = 64
    high_dynamics: bool = False
    position_encoder: nn.Module = HashGridEncoder(num_levels=16, table_size=2 ** 19,
                                                  feature_dims=2, max_resolution=2048)
    view_encoder: nn.Module = SphericalHarmonic4thEncoder()

    @nn.compact
    def __call__(self, input_points, input_views):
        input_points = self.normalize_points(input_points)
        encoded_points = self.position_encoder(input_points)

        # density MLP, two layers of ReLU
        x = nn.Dense(features=self.num_hidden_features)(encoded_points)
        x = nn.relu(x)
        x = nn.Dense(features=self.num_hidden_features)(x)
        x = nn.relu(x)
        x = nn.Dense(features=16)(x)

        # we use relu instead of exp since we do want density == 0 for empty space.
        densities = nn.relu(x[..., 0])

        # color MLP, 3 layers of ReLu
        encoded_dir = self.view_encoder(input_views)
        x = jnp.concatenate([x[..., 1:], encoded_dir], axis=-1)
        x = nn.Dense(features=self.num_hidden_features)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.num_hidden_features)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.num_hidden_features)(x)
        x = nn.relu(x)
        x = nn.Dense(features=3)(x)

        if self.high_dynamics:
            colors = jnp.exp(x)
        else:
            colors = nn.sigmoid(x)

        return colors, densities


class SirenNeRFModel(NeRF):
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
        # density MLP, 3 layers of SIREN
        x = Sine(hidden_features=self.num_hidden_features, is_first=True)(encoded_points)
        x = Sine(hidden_features=self.num_hidden_features)(x)
        x = Sine(hidden_features=self.num_hidden_features)(x)
        x = nn.Dense(features=16, kernel_init=self.init_last)(x)

        # we use relu instead of exp since we do want density == 0 for empty space.
        densities = nn.relu(x[..., 0])

        # color MLP, 2 layers of SIREN
        encoded_dir = self.view_encoder(input_views)
        x = jnp.concatenate([x[..., 1:], encoded_dir], axis=-1)

        x = Sine(hidden_features=self.num_hidden_features)(x)
        x = Sine(hidden_features=self.num_hidden_features)(x)
        x = nn.Dense(features=3, kernel_init=self.init_last)(x)

        colors = nn.sigmoid(x)
        return colors, densities


class SirenRBNeRFModel(NeRF):
    num_hidden_features: int = 64
    position_encoder: nn.Module = PositionalEncodingNeRF(num_frequencies=10)
    view_encoder: nn.Module = SphericalHarmonic4thEncoder()

    @nn.compact
    def __call__(self, input_points, input_views):
        input_points = self.normalize_points(input_points)
        encoded_points = self.position_encoder(input_points)

        x = Sine(hidden_features=self.num_hidden_features, is_first=True)(encoded_points)
        x = SirenResidualBlock(hidden_features=self.num_hidden_features,
                               out_features=16,
                               is_last=True)(x)

        # we use relu instead of exp since we do want density == 0 for empty space.
        densities = nn.relu(x[..., 0])

        encoded_dir = self.view_encoder(input_views)
        x = jnp.concatenate([x[..., 1:], encoded_dir], axis=-1)

        x = Sine(hidden_features=self.num_hidden_features)(x)
        x = SirenResidualBlock(hidden_features=self.num_hidden_features,
                               out_features=3,
                               is_last=True)(x)

        colors = nn.sigmoid(x)
        return colors, densities


class ReLuNeRFModel(nn.Module):
    num_hidden_layers: int = 4
    num_hidden_features: int = 256
    apply_positional_encoding: bool = True

    @nn.compact
    def __call__(self, inputs, *args, **kwargs):
        x = PositionalEncodingNeRF()(inputs) if self.apply_positional_encoding else inputs
        for i in range(self.num_hidden_layers):
            x = nn.Dense(features=self.num_hidden_features, dtype=float)(x)
            x = nn.relu(x)
        x = nn.Dense(features=4)(x)
        return raw2output(x)


class WireNeRFModel(nn.Module):
    num_hidden_layers: int = 4
    apply_positional_encoding: bool = False

    @nn.compact
    def __call__(self, inputs, *args, **kwargs):
        x = PositionalEncodingNeRF()(inputs) if self.apply_positional_encoding else inputs
        x = Wire(hidden_layers=self.num_hidden_layers, out_features=4)(x)
        return raw2output(x)
