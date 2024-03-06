from abc import abstractmethod
from dataclasses import dataclass

import jax.numpy as jnp
import jax.random
from jaxtyping import Array, Float, Bool


@dataclass
class PixelSampler:
    width: int
    height: int

    @abstractmethod
    def generate_samples(self, *args, **kwargs) -> Float[Array, "n_samples 2"]:
        """Generates pixel samples
        Returns
            (row, col) array of pixel coordinates
        """

    def __call__(self, *args, **kwargs):
        return self.generate_samples(*args, **kwargs)


@dataclass
class Dense(PixelSampler):
    """
    Sample all the pixels in an image of width by height.
     Returns
      [height*width, 2] of (row, col) coordinates.
    """

    def generate_samples(self):
        col, row = jnp.meshgrid(jnp.arange(self.width, dtype=jnp.float32),
                                jnp.arange(self.height, dtype=jnp.float32))

        return jnp.stack([row, col], axis=-1).reshape((-1, 2))


@dataclass
class UniformRandom(PixelSampler):
    """
    Uniform random sample of pixels in an image of width by height.
    Returns
     [n_samples, 2] of (row, col) coordinates
    """
    n_samples: int

    def generate_samples(self, rng: jax.random.PRNGKey):
        return jax.random.choice(rng, Dense(self.width, self.height)(), shape=(self.n_samples,))


@dataclass
class MaskedUniformRandom(PixelSampler):
    """
     Uniformly random sample of pixels where mask is True.
     :param mask:
     :param n_samples:  Number of samples
     :param rng: PRNG key
     :return: (N_samples, 2) of (row, col) coordinates
     """

    def generate_samples(self,
                         mask: Bool[Array, "n_rows n_cols"],
                         n_samples: int,
                         rng: jax.random.PRNGKey):
        assert (self.height, self.width) == mask.shape

        pixels = Dense(width=self.width, height=self.height)()
        cdf = jnp.cumsum(mask.reshape(-1, 1))
        indices = jnp.searchsorted(cdf,
                                   jax.random.randint(rng, shape=(n_samples,),
                                                      minval=0, maxval=cdf[-1]),
                                   side="right")
        return pixels[indices]


if __name__ == "__main__":
    coords = Dense(width=10, height=20)()
    assert coords.shape == (20 * 10, 2)

    key = jax.random.PRNGKey(123)
    samples = UniformRandom(width=20, height=10, n_samples=7)(key)
    assert samples.shape == (7, 2)

    key, _ = jax.random.split(key)
    mask = jax.random.bernoulli(key, shape=(3, 4))
    samples = MaskedUniformRandom(width=4, height=3)(mask, 5, key).astype(int)
    assert samples.shape[0] == 5
    assert mask[samples[:, 0], samples[:, 1]].all()
