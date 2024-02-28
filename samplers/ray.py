import jax
import jax.numpy as jnp


def stratified_random(ray_origins, ray_directions, t_near, t_far, n_samples, rng_key):
    """
    Stratified random sampling on each ray in the range of [t_near, t_far].

    Parameters:
        ray_origins: (num_rays, 3) array of ray origins
        ray_directions: (num_rays, 3) array of ray directions
        t_near: float, minimum distance for stratified sampling
        t_far: float, maximum distance for stratified sampling
        n_samples: number of samples to draw for each ray
        rng_key: PRNG key

    Returns:
        points: (num_rays, num_samples, 3) array of sampled points coordinates
        t_i: (num_rays, num_samples) array of sampled t values
    """
    # stratified random sampling of t in [t_near, t_far], we start with uniform
    # sampling in [t_near, t_far] and add some randomness in it.
    t_i = jnp.linspace(t_near, t_far, n_samples, axis=-1)
    # for each ray, we need to generate n_sample of random numbers, totally
    # N_pixels by N_samples of them.
    noise_shape = ray_origins.shape[:-1] + (n_samples,)

    # FIXME: do we have to do this?
    # if not jnp.isscalar(t_far):
    #     t_far = t_far.reshape((-1, 1))
    # if not jnp.isscalar(t_near):
    #     t_near = t_near.reshape((-1, 1))

    t_i += jax.random.uniform(rng_key, shape=noise_shape) * (t_far - t_near) / n_samples
    # p = r_o + t * r_d
    # multiply ray_direction of (num_rays, 3) by t_i of (num_rays, num_samples) -> (num_rays, num_samples, 3)
    points = ray_origins[..., jnp.newaxis, :] + jnp.einsum('ik,ij->ikj', t_i, ray_directions)

    return points, t_i


def importance_sampling(ray_origins, ray_directions, t_values, weights, n_samples, rng_key, combine=True):
    """
    Importance sampling for rays.

    Given (sorted) sample points `t_values` on the ray and their relative importance
     `weights`, sample more points in between in proportional to the weights.

    Note that is function does not sample areas for t < t_values[0] and t >= t_values[-1].

    Parameters:
        ray_origins: (num_rays, 3) array of origins
        ray_directions: (num_rays, 3) array of ray directions
        t_values: (num rays, num t values) locations on the rays where importance is given, assume sorted
        weights: (num rays, num t values) array of relative importance
        n_samples: number of samples to draw for each ray
        rng_key: PRNG key
        combine: whether to combine new samples with t_values

    Returns:
        points: (num_rays, num_samples, 3) array of sampled points coordinates
        t_i: (num_rays, num_samples) array of sampled t values
    """

    @jax.vmap
    def per_ray_op(r_o, r_d, ts, ws, rng):
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
                                 shape=(n_samples,))

        # Search for the *bins* where the samples are.
        indices = jnp.searchsorted(cdf, u_i, side='right')

        # Calculate the linear interpolation ratio for the samples.
        # Here ws is actually delta_cdf
        r_i = (u_i - cdf[indices]) / ws[indices]

        # Linear interpolate for samples on t axis.
        delta_t = jnp.diff(ts)
        t_i = r_i * delta_t[indices] + ts[indices]

        t_i = jax.lax.stop_gradient(t_i)

        if combine:
            t_i = jnp.concatenate((t_i, ts))

        # Sort samples in t to facilitate volume rendering.
        t_i = jnp.sort(t_i)

        points = r_o[jnp.newaxis, :] + r_d[jnp.newaxis] * t_i[:, jnp.newaxis]
        return points, t_i

    return per_ray_op(ray_origins, ray_directions, t_values, weights,
                      jax.random.split(rng_key, ray_origins.shape[0]))
