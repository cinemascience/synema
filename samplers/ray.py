from abc import abstractmethod
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from renderers.rays import RayBundle, RaySamples


@dataclass
class RaySampler:
    """
    Parameters:
        n_samples: Number of samples per ray to sample
    Returns
        ray_samples: sampled points and t-values of each ray
    """
    n_samples: int

    @abstractmethod
    def generate_samples(self, *args, **kwargs) -> Float[Array, "num_rays num_samples"]:
        """Generate n_samples of t values for each ray"""

    def __call__(self, ray_bundle: RayBundle, *args, **kwargs) -> RaySamples:
        t_values = self.generate_samples(ray_bundle, *args, **kwargs)

        # p = r_o + t * r_d
        # multiply ray_direction of (num_rays, 3) by t_i of (num_rays, num_samples) -> (num_rays, num_samples, 3)
        points = (ray_bundle.origins[..., jnp.newaxis, :] +
                  jnp.einsum('ik,ij->ikj', t_values, ray_bundle.directions))

        return RaySamples(points=points, t_values=t_values)


@dataclass
class StratifiedRandom(RaySampler):
    def generate_samples(self, ray_bundle: RayBundle, rng: jax.random.PRNGKey) -> Float[Array, "num_rays num_samples"]:
        # stratified random sampling of t in [t_near, t_far], we start with uniform
        # sampling in [t_near, t_far] and add some randomness in it.
        t_values = jnp.linspace(ray_bundle.t_nears, ray_bundle.t_fars, self.n_samples, axis=-1)

        # for each ray, we need to generate n_sample of random numbers, totally
        # N_pixels by N_samples of them.
        noise_shape = ray_bundle.origins.shape[:-1] + (self.n_samples,)

        t_values += jax.random.uniform(rng, shape=noise_shape) * (
                ray_bundle.t_fars - ray_bundle.t_nears) / self.n_samples
        return t_values


@dataclass
class Importance(RaySampler):
    """
    Importance sampling for rays.
        Given (sorted) sample points `t_values` on the ray and their relative importance `weights`,
        sample more points in between in proportional to the weights.
    Parameters:
        combine: whether to combine new samples with t_values
    """
    combine: bool = True

    def generate_samples(self,
                         ray_bundle: RayBundle,
                         rng: jax.random.PRNGKey,
                         t_values: Float[Array, "num_rays num_t_values"],
                         weights: Float[Array, "num_rays num_t_values"]) -> Float[Array, "num_rays num_samples"]:
        """
        Sample more t values. Note that is function does not sample areas for t < t_values[0]
        and t >= t_values[-1].

        Parameters:
            ray_bundle: array of origins and directions
            t_values: locations on the rays where importance is given, assume sorted
            weights: array of relative importance
            rng: PRNG key
        Returns:
            points: array of sampled points coordinates
            t_i: array of sampled t values
        """

        @jax.vmap
        def for_each_ray(ts, ws, rng):
            """
            Per-ray operations to be applied to each ray, we then vmap this operation the ray bundle.
            """
            # Making sure the sum of weights is not too close to zero (happens when there is
            # nothing in the scene), to prevent NaN when normalizing cfd.
            ws += 1e-5
            cdf = jnp.cumsum(ws, axis=-1)
            cdf = cdf / cdf[-1]

            # Uniformly sample the range of cdf. Note that it does not start with 0.
            u_i = jax.random.uniform(rng,
                                     minval=cdf[0],
                                     maxval=cdf[-1],
                                     shape=(self.n_samples,))

            # Search for the *bins* where the samples are.
            indices = jnp.searchsorted(cdf, u_i, side='right')

            # Calculate the linear interpolation ratio for the samples.
            # Here ws is actually delta_cdf
            r_i = (u_i - cdf[indices]) / ws[indices]

            # Linear interpolate for samples on t axis.
            delta_t = jnp.diff(ts)
            t_i = r_i * delta_t[indices] + ts[indices]

            t_i = jax.lax.stop_gradient(t_i)

            if self.combine:
                t_i = jnp.concatenate((t_i, ts))

            # Sort samples in t to facilitate volume rendering.
            t_i = jnp.sort(t_i)

            return t_i

        return for_each_ray(t_values, weights,
                            jax.random.split(rng, ray_bundle.origins.shape[0]))


@dataclass
class DepthGuided(RaySampler):
    """Depth-guided sampling of rays inspired by https://barbararoessle.github.io/dense_depth_priors_nerf/.
     Use given ground truth depth value to sample rays.
     Samples for background pixels are distributed between near and far planes.
     Half of the samples for foreground pixels are distributed between the near and far values.
     The second half of the samples for foreground pixels are drawn from the Gaussian distribution
     centered at the depth prior.

     Note: NaN depth values are used to signify background pixels, this might change (to be 0) in
      the future.
    """
    sigma: float = 0.01

    def generate_samples(self,
                         ray_bundle: RayBundle,
                         rng_key: jax.random.PRNGKey,
                         depth_gt: Float[Array, "num_ray"]) -> Float[Array, "num_rays num_samples"]:
        fg_key0, fg_key1, bg_key = jax.random.split(rng_key, 3)

        bg_t_values = StratifiedRandom(self.n_samples).generate_samples(ray_bundle, bg_key)

        first_half = StratifiedRandom(self.n_samples // 2).generate_samples(ray_bundle, fg_key0)
        second_half = depth_gt + jax.random.normal(key=fg_key1,
                                                   shape=first_half.shape) * self.sigma
        fg_t_values = jnp.concatenate([first_half, second_half], axis=-1)
        fg_t_values = jnp.sort(fg_t_values, axis=-1)

        t_values = jnp.where(jnp.isnan(depth_gt), bg_t_values, fg_t_values)
        return t_values
