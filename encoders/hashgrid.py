import math

import flax.linen as nn
import jax.numpy as jnp
import jax.random
from jaxtyping import Array, Float, Int32, UInt32


class HashGridEncoder(nn.Module):
    r"""
    Jax/Flax implementation of multi-resolution hash grid encoder from the Instant NGP paper.
    """

    # Number of levels (L)
    num_levels: int = 16
    # Maximum entries per level (hash table size, T), 2**16 to 2**24
    # Recommended size F * T * L = 2 ** 24
    table_size: int = 2 ** 19
    # Number of feature dimensions per entry (F), 2
    feature_dims: int = 2
    # Coarsest resolution (N_min)
    min_resolution: int = 16
    # Finest resolution (N_max), 512 to 524288
    max_resolution: int = 2 ** 19

    def setup(self):
        growth_factor = math.exp((math.log(self.max_resolution) - math.log(self.min_resolution)) /
                                 (self.num_levels - 1))

        self.resolutions = jnp.floor(self.min_resolution * growth_factor ** jnp.arange(self.num_levels))
        self.resolutions = self.resolutions.reshape((-1, 1))

        # offset to the l-th hash table at the l-level
        self.table_offsets = self.table_size * jnp.arange(self.num_levels, dtype=jnp.uint32)

        self.hash_table = self.param("hash_table",
                                     lambda key, shape: jax.random.uniform(key, shape,
                                                                           minval=-1.e-4,
                                                                           maxval=1.e-4),
                                     (self.table_size * self.num_levels, self.feature_dims))

    def hash_function(self, ijk: Int32[Array, "num_of_points 8 3"]) -> UInt32[Array, "num_of_points 8"]:
        r"""
        Spatial hash function to encode grid coordinates into hash values
        Args:
             ijk: i, j, k coordinate of grid vertices
        Returns:
             encoded: hash-encoded coordinates
        """
        primes = jnp.array([1, 2_654_435_761, 805_459_861], dtype=jnp.uint32)
        encoded = jax.lax.reduce(ijk.astype(jnp.uint32) * primes, jnp.uint32(0), jnp.bitwise_xor, dimensions=(2,))
        encoded = encoded % self.table_size
        return encoded

    def __call__(self, points: Float[Array, "num_of_points 3"]) -> \
            Float[Array, "num_of_points num_levels*feature_dims"]:
        r"""
        Args:
             points: (x, y, z) coordinate of points
        Returns:
             encoded: encoded features
        """

        def for_each_layer(vertices: Int32[Array, "num_of_points 8 3"],
                           table_offset: jnp.uint32,
                           weight: Float[Array, "num_of_points 3"]) -> \
                Float[Array, "num_of_points feature_dims"]:
            r"""
            Args:
                vertices: i, j, k coordinates of the eight vertices.
                table_offset: offset to the hash table at level
                weight: interpolation weights
            Returns:
                encoded: encoded features
            """
            hashed = self.hash_function(vertices)

            # indices of the 8 vertices to the l-th level of hash table.
            indices = hashed + table_offset

            # (num points, 8, feature_dims) features
            # FIXME: tensorboard shows this table lookup takes most of the time when doing backpropagation.
            features = self.hash_table[indices, :]

            # tri-linear interpolation
            # TODO: turn this into matrix multiplication or einsum?
            weight = weight[..., jnp.newaxis]
            one_minus_w = 1. - weight
            f_01 = features[:, 0, :] * (one_minus_w[:, 0, :]) + features[:, 1, :] * weight[:, 0, :]
            f_23 = features[:, 2, :] * (one_minus_w[:, 0, :]) + features[:, 3, :] * weight[:, 0, :]
            f_45 = features[:, 4, :] * (one_minus_w[:, 0, :]) + features[:, 5, :] * weight[:, 0, :]
            f_67 = features[:, 6, :] * (one_minus_w[:, 0, :]) + features[:, 7, :] * weight[:, 0, :]

            f_0123 = f_01 * (one_minus_w[:, 1]) + f_23 * weight[:, 1]
            f_4567 = f_45 * (one_minus_w[:, 1]) + f_67 * weight[:, 1]

            encoded = f_0123 * (one_minus_w[:, 2]) + f_4567 * weight[:, 2]

            return encoded

        # (num points, L, 3) scaled x, y, z coordinates for each level
        scaled = jax.vmap(lambda res: points * res + 0.5, out_axes=1)(self.resolutions)
        # (num points, L, 3) i, j, k indices of the first vertex.
        floors = jnp.floor(scaled).astype(jnp.int32)
        weights = scaled - floors

        vertices_of_cell = jnp.asarray([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
        ])

        # (num points, L, 8, 3), i, j, k coordinates of the 8 vertices where the input
        # point falls to. There are L levels of grids.
        all_vertices = floors[:, :, jnp.newaxis, :] + vertices_of_cell[jnp.newaxis, jnp.newaxis, :]

        encoded_vertices = jax.vmap(for_each_layer,
                                    in_axes=(1, 0, 1),
                                    out_axes=1)(all_vertices, self.table_offsets, weights)
        return encoded_vertices.reshape(points.shape[0], -1)


if __name__ == "__main__":
    x = jnp.array([1.1, 2.2, 3.3]).reshape((-1, 3))

    key = jax.random.PRNGKey(0)
    encoder = HashGridEncoder()
    params = encoder.init(key, x)

    encoded = encoder.apply(params, x)
    print(encoded)
