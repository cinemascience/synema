from abc import abstractmethod
from dataclasses import dataclass
from typing import Callable

import jax.numpy as jnp
import jax.random
import jax.random
from jaxtyping import Array, Float

import synema.samplers.ray
from synema.renderers.rays import RayBundle
from synema.samplers.ray import StratifiedRandom, Importance


@dataclass
class VolumeRenderer:
    @staticmethod
    def sample_radiance_field(field_fn: Callable,
                              points: Float[Array, "num_rays num_samples_per_ray 3"],
                              viewdirs: Float[Array, "num_rays 3"] = None) -> \
            (Float[Array, "num_rays num_samples_per_ray n_channels"], Float[Array, "num_rays num_samples_per_ray"]):
        """Sample radiance field
        Parameters:
            field_fn: field function mapping from positions and view directions to colors and densities
            points: points on rays to sample the field
            viewdirs: normalized directional vector of rays
        Return:
            (colors, density) color and density predicted by the field function.
        """
        # vmap vectorize along the num_rays dimension, we also add another dimension for num_samples
        # for viewdirs.
        return jax.vmap(field_fn)(points, jnp.broadcast_to(viewdirs[:, None, :], points.shape))

    @staticmethod
    def sample_radiance_field_batch(field_fn: Callable,
                                    points: Float[Array, "num_rays num_samples_per_ray 3"],
                                    viewdirs: Float[Array, "num_rays 3"] = None,
                                    batch_size: int = 4096):
        # Note: This only fixes the OOM problem for inference.
        # It does not help during training since we need to batch pixels before
        # they are sent to the loss function (which is reverse-mode auto-diffed).
        x = [VolumeRenderer.sample_radiance_field(field_fn,
                                                  points[i:i + batch_size],
                                                  viewdirs[i:i + batch_size])
             for i in range(0, points.shape[0], batch_size)]
        return (jnp.concat([x[i][0] for i in range(len(x))]),
                jnp.concat([x[i][1] for i in range(len(x))]))

    # Compute weight used for quadrature for the volume rendering. First we compute
    # the accumulated transmittance T_i = exp(-sum(sigma_i delta_i))
    # where delta_i = t_{i+1} - t_i for i = [0, N-1]. The summation inside exp()
    # is turned into product of exp(sigma_i delta_i), thus T_i = prod(exp(-sigma_i delta_i)).
    # We then compute the weights as element-wise product of T_i and alpha_i.
    # See Eqn. 3 in the original paper on arXiv. Note that the weights are NOT normalized,
    # i.e. sum(w_i) != 1 and could be very small for background pixels.
    @staticmethod
    def compute_weights(densities: Float[Array, "num_pixels num_sample_per_ray"],
                        t_vals: Float[Array, "num_pixels num_sample_per_ray"]) -> \
            Float[Array, "num_pixels num_sample_per_ray"]:
        delta_t = jnp.diff(t_vals)
        # add one extra (huge) delta_N = t_{N+1} - t_N that does not
        # exist to work with the shape of sampled densities (but
        # provide negligible contribution).
        delta_t = jnp.concatenate([delta_t,
                                   jnp.broadcast_to(1e10, delta_t[..., :1].shape)],
                                  axis=-1)
        alphas = 1. - jnp.exp(-densities * delta_t)
        # TODO: we probably don't need this clipping. exp(-x) for x >= 0 will be in [1, 0]
        clipped_densities = jnp.clip(1.0 - alphas, 0., 1.0)
        # jnp.cumprod is inclusive scan, we need to add the first 1s ourselves.
        transmittance = jnp.cumprod(jnp.concatenate([jnp.ones_like(clipped_densities[..., :1]),
                                                     clipped_densities[..., :-1]], axis=-1),
                                    axis=-1)
        # Note that the accumulated transmittance is monotonically decreasing w.r.t t_vals, thus
        # the contribution of alphas (and thus densities/sigmas) to the final weights also
        # decreases w.r.t. t_vals.
        # This achieve a similar effect as the "first hit" for ray marching when the model is
        # trained. In that case, the density function approximates delta functions located at
        # the surfaces of the scene. The first "intersection" will dominate the accumulated
        # transmittance while later intersections contributes close to zero to the transmittance.
        # Thus we will have large weight correspond to the first hit and close to zero weight
        # correspond to later hits. This explains why we don't see a blending of front and back
        # surfaces for a close surface like sphere (for both color and depth values) in the
        # final rendered images.
        return alphas * transmittance

    @staticmethod
    def sample_rays(ray_sampler, field_fn: Callable, ray_bundle: RayBundle, rng: jax.random.PRNGKey,
                    *args, **kwargs):
        ray_samples = ray_sampler(ray_bundle, rng, *args, **kwargs)

        viewdirs = ray_bundle.directions / jnp.linalg.norm(ray_bundle.directions, axis=-1, keepdims=True)

        colors, densities = VolumeRenderer.sample_radiance_field_batch(field_fn, ray_samples.points, viewdirs)
        weights = VolumeRenderer.compute_weights(densities, ray_samples.t_values)
        return colors, weights, ray_samples.t_values

    @staticmethod
    def accumulate_samples(colors, weights, t_values):
        # Einstein's notation to compute weighted average of colors, i.e., sum(weights * colors)
        # reducing (num_rays, num_samples) x (num_rays, num_samples, rgb) -> (num_rays, rgb)
        rgb = jnp.einsum('ij,ijk->ik', weights, colors)
        # Similar to the above, but reducing (num_rays, num_samples) x (num_rays, num_samples) -> (num_rays)
        depth = jnp.einsum('ij,ij->i', weights, t_values)
        # accumulated weights become the alpha channel
        alpha = jnp.einsum('ij->i', weights)

        return {'rgb': rgb, 'alpha': alpha, 'depth': depth}

    @abstractmethod
    def render(self, *args, **kwargs) -> \
            (Float[Array, "num_rays 3"], Float[Array, "num_rays"], Float[Array, "num_rays num_sample_per_ray"]):
        """Render the field, return color, alpha and depth"""

    def __call__(self, *args, **kwargs):
        return self.render(*args, **kwargs)


@dataclass
class Simple(VolumeRenderer):
    ray_sampler = StratifiedRandom(n_samples=64)

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
        coarse_key, fine_key = jax.random.split(rng_key, 2)
        # Sample and render with the coarse model
        colors, weights, t_values = self.sample_rays(self.coarse_sampler,
                                                     coarse_field,
                                                     ray_bundle,
                                                     coarse_key,
                                                     *args, **kwargs)
        coarse_output = self.accumulate_samples(colors, weights, t_values)

        # Sample and render the fine model
        colors, weights, t_values = self.sample_rays(self.fine_sampler, fine_field, ray_bundle,
                                                     fine_key, t_values=t_values, weights=weights)
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
    ray_sampler = synema.samplers.ray.DepthGuided(n_samples=64)

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
        coarse_key, fine_key = jax.random.split(rng_key)
        _, weights, t_values = self.sample_rays(ray_sampler=self.coarse_sampler,
                                                field_fn=field_fn,
                                                ray_bundle=ray_bundle,
                                                rng=coarse_key,
                                                *args, **kwargs)
        # t_mean is the expected value of t_values
        t_mean = jnp.einsum('ij,ij->i', weights, t_values)
        # t_variance is the expected value of (t_value - t_mean) ** 2
        t_variance = jnp.einsum('ij,ij->i', weights, (t_values - t_mean[:, None]) ** 2)
        t_std = jnp.sqrt(t_variance)

        # Sample the second half according to N(t_mean, t_std) for +- 3 t_std
        t_values = jnp.linspace(t_mean - 3 * t_std,
                                t_mean + 3 * t_std,
                                self.fine_sampler.n_samples,
                                axis=-1)
        t_values = jnp.clip(t_values, ray_bundle.t_nears, ray_bundle.t_fars)

        weights = jax.scipy.stats.norm.pdf(t_values, loc=t_mean[:, None], scale=t_std[:, None])

        colors, weights, t_values = self.sample_rays(ray_sampler=self.fine_sampler,
                                                     field_fn=field_fn,
                                                     ray_bundle=ray_bundle,
                                                     rng=fine_key,
                                                     t_values=t_values,
                                                     weights=weights)

        return self.accumulate_samples(colors, weights, t_values)
