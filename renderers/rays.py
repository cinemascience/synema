from flax import struct
from jaxtyping import Array, Float


# annotated with flax.struct.dataclass so it can be transformed
# by jax.jit etc.
@struct.dataclass
class RayBundle:
    origins: Float[Array, "num_rays 3"]
    directions: Float[Array, "num_rays 3"]
    # TODO: should t_nears/t_far be here? Should it be input to RaySamplers?
    t_nears: Float[Array, "num_rays"]
    t_fars: Float[Array, "num_rays"]


@struct.dataclass
class RaySamples:
    points: Float[Array, "num_rays num_samples_per_ray 3"]
    t_values: Float[Array, "num_rays num_samples_per_ray"]
