import flax.linen as nn
import jax.numpy as jnp
from jaxtyping import Array, Float


class SphericalHarmonic4thEncoder(nn.Module):
    r"""Fourth order spherical harmonic encoding for view directions"""

    @nn.compact
    def __call__(self, dirs: Float[Array, "num_rays 3"]) -> Float[Array, "num_rays 16"]:
        """
        Args:
            dirs: normalized directional vectors for rays
        Returns:
            encoding: encoded ray directional vectors
        """
        x, y, z = dirs[..., 0], dirs[..., 1], dirs[..., 2]
        xy, xz, yz = x * y, x * z, y * z
        xx, yy, zz = x ** 2, y ** 2, z ** 2

        encoding = jnp.array([
            0.28209479177387814 * jnp.ones_like(x),
            0.4886025119029199 * y,
            0.4886025119029199 * z,
            0.4886025119029199 * x,
            1.0925484305920792 * xy,
            1.0925484305920792 * yz,
            0.9461746957575601 * zz - 0.31539156525251999,
            1.0925484305920792 * xz,
            0.5462742152960396 * (xx - yy),
            0.5900435899266435 * y * (3. * xx - yy),
            2.890611442640554 * x * y * z,
            0.4570457994644658 * y * (5. * zz - 1.),
            0.3731763325901154 * z * (5. * zz - 3.),
            0.4570457994644658 * x * (5. * zz - 1.),
            1.445305721320277 * z * (xx - yy),
            0.5900435899266435 * x * (xx - 3. * yy)]).T

        return encoding
