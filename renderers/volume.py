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

        viewdirs = ray_bundle.directions / jnp.linalg.norm(ray_bundle.directions)

        colors, opacities = VolumeRenderer.sample_radiance_field(field_fn, ray_samples.points, viewdirs)
        weights = VolumeRenderer.accumulated_transmittance(opacities, ray_samples.t_values)
        return colors, weights, ray_samples.t_values

    @staticmethod
    def accumulate_samples(colors, weights, t_values):
        # Einstein's notation to compute weighted ave0rage of colors, i.e., sum(weights * colors)
        # reducing (num_rays, num_samples) x (num_rays, num_samples, rgb) -> (num_rays, rgb)
        rgb = jnp.einsum('ij,ijk->ik', weights, colors)
        # Similar to the above, but reducing (num_rays, num_samples) x (num_rays, num_samples) -> (num_rays)
        depth = jnp.einsum('ij,ij->i', weights, t_values)
        # accumulated opacities become the alpha channel
        alpha = jnp.einsum('ij->i', weights)

        # FIXME: Is this the right way to calculate depth?
        # depth = -depth * jnp.linalg.norm(ray_directions, axis=-1)
        return rgb, alpha, depth

    @abstractmethod
    def render(self, *args, **kwargs) -> \
            (Float[Array, "num_rays 3"], Float[Array, "num_rays"], Float[Array, "num_rays num_sample_per_ray"]):
        """Render the field, return color, alpha and depth"""

    def __call__(self, *args, **kwargs):
        return self.render(*args, **kwargs)


@dataclass
class Simple(VolumeRenderer):
    # TODO: why type annotation doesn't work here?
    ray_sampler = StratifiedRandom(n_samples=32)

    def render(self,
               field_fn: Callable,
               ray_bundle: RayBundle,
               rng_key: jax.random.PRNGKey):
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
        colors_coarse, _, _ = self.accumulate_samples(colors, weights, t_values)

        # Sample and render the fine model
        rng_key, _ = jax.random.split(rng_key)
        colors, weights, t_values = self.sample_rays(self.fine_sampler, fine_field, ray_bundle,
                                                     rng_key, t_values=t_values, weights=weights)
        colors_fine, alpha, depths = self.accumulate_samples(colors, weights, t_values)

        return (colors_coarse, colors_fine), alpha, depths


@dataclass
class DepthGuided(Hierarchical):
    """Depth-guided renderer inspired by https://barbararoessle.github.io/dense_depth_priors_nerf/.
    Replace the coarse sampler with DepthGuided and use one single field model.
    """
    coarse_sampler = samplers.ray.DepthGuided(n_samples=64)

    def render(self,
               field_fn: Callable,
               ray_bundle: RayBundle,
               rng_key: jax.random.PRNGKey,
               depth_gt: Float[Array, "num_rays num_samples"],
               *args, **kwargs):
        return super().render(field_fn, field_fn,
                              ray_bundle, rng_key,
                              depth_gt,
                              *args, **kwargs)


@dataclass
class DepthGuidedInfer(DepthGuided):
    # TODO: unfinished.
    coarse_sampler = StratifiedRandom(n_samples=32)
    fine_sampler = Importance(n_samples=32, combine=False)

    def render(self,
               field_fn: Callable,
               ray_bundle: RayBundle,
               rng_key: jax.random.PRNGKey,
               *args, **kwargs):
        # The first half of the samples are used to render an approximate depth value
        # t_mean and standard deviation t_std
        _, _, t_values = self.sample_rays(self.coarse_sampler,
                                          field_fn,
                                          ray_bundle,
                                          rng_key,
                                          *args, **kwargs)
        t_mean = jnp.mean(t_values, axis=-1)
        t_std = jnp.std(t_values, axis=-1)

        # Sample the second half according to N(t_mean, t_std)
        # We first generate
        t_values = jnp.linspace(jax.scipy.stats.norm.ppf(0.05),
                                jax.scipy.stats.norm.ppf(0.95),
                                self.fine_sampler.n_samples)
        weights = jax.scipy.stats.norm.pdf(t_values)  # , loc=t_mean, scale=t_std)

        rng_key, _ = jax.random.split(rng_key)
        colors, weights, t_values = self.sample_rays(self.fine_sampler, field_fn, ray_bundle,
                                                     rng_key, t_values=t_values, weights=weights)

        return self.accumulate_samples(colors, weights, t_values)

# TODO: unfinished ideas, separate functions or just a single function would work?
# def sample_rgb_field(field_fn, points):
#     pass
#
#
# def sample_depth_field(field_fn, points):
#     shape = points.shape
#     # flatten [N_rays, N_samples, 3] into [total number of samples, 3].
#     points_flatten = points.reshape((-1, 3))
#     # return [N_rays, N_samples] of probability density
#     return field_fn(points_flatten).reshape(shape[:2])
#
#
# def march_ray_training_background(model_fn, ray_origins, ray_directions, t_near, t_far, rng):
#     """
#     March rays from z_near to z_far for background pixels. Sample model_fn along the way.
#      Return the line integral of the samples alone the rays.
#     :param model_fn:
#     :param ray_origins:
#     :param ray_directions:
#     :param t_near:
#     :param t_far:
#     :param rng:
#     :return:
#     """
#
#     # number of sample on the ray
#     n_samples_bg = 32
#
#     # 1. Stratified uniform sampling on the ray.
#     points, t_vals = stratified_random(ray_origins, ray_directions,
#                                        t_near, t_far, n_samples_bg, rng)
#     # 2. Query the model_fn for pdf(z)
#     sdf = sample_depth_field(model_fn, points)
#     # 3. Integrate pdf(z), ideally it should be 0 for BG pixels.
#     weights = nn.sigmoid(sdf) * nn.sigmoid(-sdf)
#     return jnp.sum(weights, axis=-1)
#
#
# def march_ray_training_foreground(model_fn, ray_origin, ray_direction, z_gt, rng):
#     """
#     We march rays from z_near to some pre-defined truncation_distance beyond z_gt.
#     We will sample coarsely (with stratified uniform) in "known empty" space before
#     hitting the know surface i.e. [0, z_gt - z_tr] and finely (with Gaussian) between
#     [z_gt - z_tr, z_gt + z_tr].
#     """
#
#     # TODO: hyper-parameterized it
#     z_tr = 0.01
#     n_samples_empty = 8
#     n_samples_surface = 16
#
#     # 1a. Use the ray sampler to sample the ray in the empty space before hitting
#     #  the surface.
#     empty_points, empty_zs = stratified_random(ray_origin, ray_direction,
#                                                -3.4314790205172763, z_gt + z_tr,
#                                                n_samples_empty, rng)
#     # 1b. Sample around the surface uniformly, for the moment, might change to
#     #  gaussian or triangular.
#     new_rng, _ = jax.random.split(rng)  # TODO: does this actually work?
#     surface_points, surface_zs = stratified_random(ray_origin, ray_direction,
#                                                    z_gt + z_tr, z_gt - z_tr,
#                                                    n_samples_surface, new_rng)
#
#     # 2. Given the sampled points, apply the network function to get pdf(z)
#     points = jnp.hstack((empty_points, surface_points))
#
#     zs = jnp.hstack((empty_zs, surface_zs))
#
#     # TODO: this way of computing density/weight is still wrong.
#     sdf = sample_depth_field(model_fn, points)
#     weights = sdf2weights(sdf, z_tr, zs)
#
#     # sdf = sdf / jnp.sum(sdf, axis=-1).reshape((-1,1))
#     # jax.debug.print("min: {x}", x=pdf.min())
#     # jax.debug.print("max: {x}", x=pdf.max())
#     # 3. Integrate z * pdf(z) to get the expected value of z for the pixel, this
#     # should be cloe to the GT z-image after training.
#     # jax.debug.print("min z: {x}", x=zs.min())
#     # jax.debug.print("max z: {x}", x= zs.max())
#     return jnp.sum(zs * weights, axis=-1)
#
#
# def sdf2weights(sdf, z_tr, zs):
#     weights = jax.nn.sigmoid(sdf / z_tr) * jax.nn.sigmoid(-sdf / z_tr)
#
#     # Find where the sign of SDF changes.
#     sign = sdf[..., 1:] * sdf[..., :-1]
#     # 1 at the sign change, 0 otherwise. This essentially turn the SDF into
#     # a delta function
#     mask = jnp.where(sign < 0., jnp.ones_like(sign), jnp.zeros_like(sign))
#     # find the index where the sign changes
#     indices = jnp.argmax(mask, axis=-1)
#     # get the corresponding z values, this is the z_min where the ray hits
#     # the surface
#     # z_min = jnp.take(zs, indices[..., jnp.newaxis])
#     z_min = jnp.take_along_axis(zs, indices[..., None], axis=-1)
#     mask = jnp.where(zs > z_min - z_tr, jnp.ones_like(zs), jnp.zeros_like(zs))
#     # weights are cleared to zero for z values `z_tr` away behind the surface
#     weights = weights * mask
#     return weights / (jnp.sum(weights, axis=-1, keepdims=True) + 1.e-8)
#
#
# def march_ray_inference(model_fn, ray_origins, ray_directions,
#                         z_near, z_far, rng):
#     """
#     Since we don't have prior knowledge of where the ray might hit the surface, we use
#     the two level sampling as in the original paper. We first perform a coarse stratified
#     uniform sampling, this will give as an estimate o the pdf of the z-value. We then
#     calculate the cdf and use it to sample where pdf is large.
#
#     TODO: how are we sure the pdf is single modal? Does it matter?
#     :return:
#     """
#
#     # number of sample on the ray
#     n_samples_bg = 64
#
#     # 1. Stratified uniform sampling on the ray.
#     points, z_vals = stratified_random(ray_origins, ray_directions,
#                                        z_near, z_far, n_samples_bg, rng)
#     # 2. Query the model_fn for pdf(z)
#     sdf = sample_depth_field(model_fn, points)
#     weights = sdf2weights(sdf, 0.05, z_vals)
#     weights = weights / jnp.sum(weights, axis=-1).reshape((-1, 1))
#     # jax.debug.print("min: {x}", x=pdf.min())
#     # jax.debug.print("max: {x}", x=pdf.max())
#     # jax.debug.print("min z: {x}", x=z_vals.min())
#     # jax.debug.print("max z: {x}", x=z_vals.max())
#     # jax.debug.print("exp z: {x}", x=jnp.sum(z_vals * pdf, axis=-1))
#     # 3. Integrate pdf(z), ideally it should be 0 for BG pixels.
#     return jnp.sum(z_vals * weights, axis=-1)
