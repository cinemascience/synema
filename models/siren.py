import flax.linen as nn
import jax.numpy as jnp
import jax.random


class Sine(nn.Module):
    hidden_features: int = 256
    omega_0: float = 30.
    is_first: bool = False

    def init_weights(self, key, shape, dtype):
        v = 1. / shape[0] if self.is_first else jnp.sqrt(6. / shape[0]) / self.omega_0
        return jax.random.uniform(key=key, shape=shape, dtype=dtype,
                                  minval=-v, maxval=v)

    @nn.compact
    def __call__(self, inputs, *args, **kwargs):
        x = nn.Dense(features=self.hidden_features,
                     kernel_init=self.init_weights)(inputs)
        return jnp.sin(self.omega_0 * x)


class Siren(nn.Module):
    hidden_features: int = 256
    omega_0: float = 30.
    hidden_layers: int = 4
    out_features: int = 1

    def init_last(self, key, shape, dtype):
        v = jnp.sqrt(6. / shape[0]) / self.omega_0
        return jax.random.uniform(key=key, shape=shape, dtype=dtype,
                                  minval=-v, maxval=v)

    @nn.compact
    def __call__(self, inputs, *args, **kwargs):
        x = Sine(hidden_features=self.hidden_features, omega_0=self.omega_0,
                 is_first=True)(inputs)
        for i in range(self.hidden_layers - 1):
            x = Sine(hidden_features=self.hidden_features, omega_0=self.omega_0)(x)
        return nn.Dense(features=self.out_features, kernel_init=self.init_last)(x)


if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    x = jnp.ones((10, 2))

    sine = Sine(hidden_features=16, is_first=True)
    params = sine.init(key, x)
    print(sine.tabulate(key, x))

    siren = Siren()
    params1 = siren.init(key, x)
    print(siren.tabulate(key, x))
