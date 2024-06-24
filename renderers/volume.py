from abc import abstractmethod
from dataclasses import dataclass
from typing import Callable

import jax.numpy as jnp
import jax.random
import jax.random
from jaxtyping import Array, Float

import samplers.ray
from renderers.rays import RayBundle
from samplers.ray import StratifiedRandom, Importance


@dataclass
class VolumeRenderer:
    @staticmethod
    def sample_radiance_field(field_fn: Callable,
                              points: Float[Array, "num_rays num_samples_per_ray 3"],
                              viewdirs: Float[Array, "num_rays 3"] = None) -> \
            (Float[Array, "num_rays num_samples_per_ray n_channels"], Float[Array, "num_rays num_samples_per_ray"]):
        """Sample radiance field
        Parameters:
            field_fn: field function mapping from position and view direction to color and density
            points: points on rays to sample the field
            viewdirs: normalized directional vector of rays
        Return:
            (colors, density) color and opacity predicted by the field function.
        """

        # vmap vectorize along the num_rays dimension, we also add another dimension for num_samples
        # for viewdirs.
        return jax.vmap(field_fn)(points, jnp.broadcast_to(viewdirs[:, None, :], points.shape))

    # Compute the accumulated transmittance T_i = exp(-sum(sigma_i delta_i))
    # where delta_i = t_{i+1} - t_i for i = [0, N-1].
    @staticmethod
    def accumulated_transmittance(opacities: Float[Array, "num_pixels num_sample_per_ray"],
                                  t_vals: Float[Array, "num_pixels num_sample_per_ray"]) -> \
            Float[Array, "num_pixels num_sample_per_ray"]:
        delta_t = jnp.diff(t_vals)
        # add one extra (huge) delta_N = t_{N+1} - t_N that does not
        # exist to work with the shape of sampled opacities (but
        # provide negligible contribution).
        delta_t = jnp.concatenate([delta_t,
                                   jnp.broadcast_to(1e10, delta_t[..., :1].shape)],
                                  axis=-1)
        # TODO: some implementation multiplies delta_t with the L2 norm of ray direction.
        alphas = 1. - jnp.exp(-opacities * delta_t)
        # TODO: we probably don't need this clipping. exp(-x) for x >= 0 will be in [1, 0]
        clipped_densities = jnp.clip(1.0 - alphas, 0., 1.0)
        # jnp.cumprod is inclusive scan, we need to add the first 1s ourselves.
        transmittance = jnp.cumprod(jnp.concatenate([jnp.ones_like(clipped_densities[..., :1]),
                                                     clipped_densities[..., :-1]], axis=-1),
                                    axis=-1)
        return alphas * transmittance

    @staticmethod
    def sample_rays(ray_sampler, field_fn: Callable, ray_bundle: RayBundle, rng: jax.random.PRNGKey,
                    *args, **kwargs):
        ray_samples = ray_sampler(ray_bundle, rng, *args, **kwargs)

        viewdirs = ray_bundle.directions / jnp.linalg.norm(ray_bundle.directions, axis=-1, keepdims=True)

        colors, opacities = VolumeRenderer.sample_radiance_field(field_fn, ray_samples.points, viewdirs)
        weights = VolumeRenderer.accumulated_transmittance(opacities, ray_samples.t_values)
        return colors, weights, ray_samples.t_values

    @staticmethod
    def accumulate_samples(colors, weights, t_values):
        # Einstein's notation to compute weighted average of colors, i.e., sum(weights * colors)
        # reducing (num_rays, num_samples) x (num_rays, num_samples, rgb) -> (num_rays, rgb)
        rgb = jnp.einsum('ij,ijk->ik', weights, colors)
        # Similar to the above, but reducing (num_rays, num_samples) x (num_rays, num_samples) -> (num_rays)
        depth = jnp.einsum('ij,ij->i', weights, t_values)
        # accumulated opacities become the alpha channel
        alpha = jnp.einsum('ij->i', weights)

        # FIXME: Is this the right way to calculate depth?
        # depth = -depth * jnp.linalg.norm(ray_directions, axis=-1)
        return {'rgb': rgb, 'alpha': alpha, 'depth': depth}

    @abstractmethod
    def render(self, *args, **kwargs) -> \
            (Float[Array, "num_rays 3"], Float[Array, "num_rays"], Float[Array, "num_rays num_sample_per_ray"]):
        """Render the field, return color, alpha and depth"""

    def __call__(self, *args, **kwargs):
        return self.render(*args, **kwargs)


@dataclass
class Simple(VolumeRenderer):
    # TODO: why type annotation doesn't work here?
    # TODO: why is TinyNeRFNodel sensitive on n_samples?
    ray_sampler = StratifiedRandom(n_samples=16)

    def render(self,
               field_fn: Callable,
               ray_bundle: RayBundle,
               rng_key: jax.random.PRNGKey,
               *args, **kwargs):
        colors, weights, t_values = self.sample_rays(self.ray_sampler, field_fn, ray_bundle, rng_key)
        return self.accumulate_samples(colors, weights, t_values)


@dataclass
class Hierarchical(VolumeRenderer):
    coarse_sampler = StratifiedRandom(n_samples=64)
    fine_sampler = Importance(n_samples=128)

    def render(self,
               coarse_field: Callable,
               fine_field: Callable,
               ray_bundle: RayBundle,
               rng_key: jax.random.PRNGKey,
               *args, **kwargs):
        # Sample and render with the coarse model
        colors, weights, t_values = self.sample_rays(self.coarse_sampler,
                                                     coarse_field,
                                                     ray_bundle,
                                                     rng_key,
                                                     *args, **kwargs)
        coarse_output = self.accumulate_samples(colors, weights, t_values)

        # Sample and render the fine model
        rng_key, _ = jax.random.split(rng_key)
        colors, weights, t_values = self.sample_rays(self.fine_sampler, fine_field, ray_bundle,
                                                     rng_key, t_values=t_values, weights=weights)
        fine_output = self.accumulate_samples(colors, weights, t_values)

        return {'coarse_rgb': coarse_output['rgb'],
                'fine_rgb': fine_output['rgb'],
                'alpha': fine_output['alpha'],
                'depth': fine_output['depth']}


@dataclass
class DepthGuidedTrain(VolumeRenderer):
    """Depth-guided renderer (for training, when ground truth depth is available) inspired by
     https://barbararoessle.github.io/dense_depth_priors_nerf/.
    """
    ray_sampler = samplers.ray.DepthGuided(n_samples=64)

    def render(self,
               field_fn: Callable,
               ray_bundle: RayBundle,
               rng_key: jax.random.PRNGKey,
               depth_gt: Float[Array, "num_rays num_samples"],
               *args, **kwargs):
        colors, weights, t_values = self.sample_rays(self.ray_sampler,
                                                     field_fn,
                                                     ray_bundle, rng_key,
                                                     depth_gt,
                                                     *args, **kwargs)
        return self.accumulate_samples(colors, weights, t_values)


@dataclass
class DepthGuidedInfer(VolumeRenderer):
    """Depth-guided renderer (for inference, where ground truth depth value is missing) inspired by
     https://barbararoessle.github.io/dense_depth_priors_nerf/.
    """
    coarse_sampler = StratifiedRandom(n_samples=64)
    fine_sampler = Importance(n_samples=64, combine=True)

    def render(self,
               field_fn: Callable,
               ray_bundle: RayBundle,
               rng_key: jax.random.PRNGKey,
               *args, **kwargs):
        # The first half of the samples are used to render an approximate depth value
        # t_mean and standard deviation t_std
        _, _, t_values = self.sample_rays(ray_sampler=self.coarse_sampler,
                                          field_fn=field_fn,
                                          ray_bundle=ray_bundle,
                                          rng=rng_key,
                                          *args, **kwargs)
        t_mean = jnp.mean(t_values, axis=-1)
        t_std = jnp.std(t_values, axis=-1)

        # Sample the second half according to N(t_mean, t_std)
        # TODO: we could just turn this to use jax.random.normal in yet another RaySampler
        t_values = jnp.linspace(jax.scipy.stats.norm.ppf(0.05, loc=t_mean, scale=t_std),
                                jax.scipy.stats.norm.ppf(0.95, loc=t_mean, scale=t_std),
                                self.fine_sampler.n_samples,
                                axis=-1)
        weights = jax.scipy.stats.norm.pdf(t_values, loc=t_mean[:, None], scale=t_std[:, None])
        rng_key, _ = jax.random.split(rng_key)
        colors, weights, t_values = self.sample_rays(ray_sampler=self.fine_sampler,
                                                     field_fn=field_fn,
                                                     ray_bundle=ray_bundle,
                                                     rng=rng_key,
                                                     t_values=t_values,
                                                     weights=weights)

        return self.accumulate_samples(colors, weights, t_values)
