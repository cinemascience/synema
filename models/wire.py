import flax.linen as nn
import jax.numpy as jnp


class Wavelet(nn.Module):
    hidden_features: int = 256
    omega_0: float = 30.0
    s_0: float = 10.0

    @nn.compact
    def __call__(self, inputs, *args, **kwargs):
        freqs = nn.Dense(features=self.hidden_features)(inputs)
        scale = nn.Dense(features=self.hidden_features)(inputs)

        return jnp.cos(self.omega_0 * freqs) * jnp.exp(-jnp.square(self.s_0 * scale))


class Wire(nn.Module):
    hidden_features: int = 256
    omega_0: float = 30.0
    s_0: float = 10.0
    hidden_layers: int = 4
    out_features: int = 1

    @nn.compact
    def __call__(self, inputs, *args, **kwargs):
        for _ in range(self.hidden_layers):
            inputs = Wavelet(hidden_features=self.hidden_features,
                             omega_0=self.omega_0,
                             s_0=self.s_0)(inputs)
        return nn.Dense(features=self.out_features)(inputs)
