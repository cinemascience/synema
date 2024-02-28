import jax.numpy as jnp
import jax.random
from dataclasses import dataclass


@dataclass
class Dense:
    """
    Sample all the pixels in an image of width by height. Return [height*width, 2] of (row, col) coordinates.
    """
    width: int
    height: int

    def __call__(self, *args, **kwargs):
        col, row = jnp.meshgrid(jnp.arange(self.width, dtype=jnp.float32),
                                jnp.arange(self.height, dtype=jnp.float32))

        return jnp.stack([row, col], axis=-1).reshape((-1, 2))


@dataclass
class UniformRandom:
    """
    Uniform random sample of pixels in an image of width by height. Return [n_samples, 2] of (row, col) coordinates
    """
    width: int
    height: int
    n_samples: int

    def __call__(self, rng, *args, **kwargs):
        return jax.random.choice(rng, Dense(self.width, self.height)(), shape=(self.n_samples,))


class MaskedUniformRandom:
    def __call__(self, mask, n_samples, rng, *args, **kwargs):
        """
        Uniformly random sample of pixels where mask is True.
        :param mask:
        :param n_samples:  Number of samples
        :param rng: PRNG key
        :param args:
        :param kwargs:
        :return: (N_samples, 2) of (row, col) coordinates
        """

        height, width = mask.shape
        pixels = Dense(width=width, height=height)()
        cdf = jnp.cumsum(mask.reshape(-1, 1))
        indices = jnp.searchsorted(cdf,
                                   jax.random.randint(rng, shape=(n_samples,),
                                                      minval=0, maxval=cdf[-1]),
                                   side="right")
        return pixels[indices]


if __name__ == "__main__":
    coords = Dense(10, 20)()
    assert coords.shape == (20 * 10, 2)

    key = jax.random.PRNGKey(123)
    samples = UniformRandom(20, 10, 7)(key)
    assert samples.shape == (7, 2)

    key, _ = jax.random.split(key)
    mask = jax.random.bernoulli(key, shape=(3, 4))
    samples = MaskedUniformRandom()(mask, 5, key).astype(int)
    assert samples.shape[0] == 5
    assert mask[samples[:, 0], samples[:, 1]].all()
