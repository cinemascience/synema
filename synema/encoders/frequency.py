import flax.linen as nn
import jax.numpy as jnp
import jax.random
from jax.nn.initializers import normal
from jaxtyping import Array, Float


class PositionalEncodingNeRF(nn.Module):
    """Positional encoding as in the original NeRF paper

    Args:
        num_frequencies: number of frequencies to encode input coordinates
    """
    num_frequencies: int = 10

    @nn.compact
    def __call__(self, inputs: Float[Array, "num_of_points 3"]) -> Float[
        Array, "num_of_points num_frequencies*2"]:
        """
        Args: inputs, coordinates of points, for best downstream quality,
         should be normalized to the range of [0, 1].
        """
        # The original paper claimed to multiply input with PI before sending
        # to sin/cos.
        # However, the real implementation does not do so.
        # inputs = inputs * 2. * jnp.pi

        pos = jnp.hstack(jax.vmap(lambda freq: inputs * 2.0 ** freq)(jnp.arange(self.num_frequencies)))
        return jnp.hstack((jnp.sin(pos), jnp.cos(pos)))


class RandomFourierFeatures(nn.Module):
    num_frequencies: int = 10
    scale: float = 10.0

    @nn.compact
    def __call__(self, inputs, *args, **kwargs):
        b_matrix = self.param("b_matrix", normal(self.scale),
                              (inputs.shape[-1], self.num_frequencies))
        inputs = inputs * jnp.pi
        x = jnp.dot(inputs, b_matrix)
        return jnp.hstack([jnp.sin(x), jnp.cos(x)])


if __name__ == "__main__":
    x = jnp.arange(15).reshape((-1, 3))
    print(x)

    key = jax.random.PRNGKey(0)
    encoder = PositionalEncodingNeRF()
    params = encoder.init(key, x)
    # print(params)
    # print(encoder.apply(params, x).shape)
    print(encoder.apply(params, x))
