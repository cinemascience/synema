from abc import abstractmethod
from dataclasses import dataclass

import jax.numpy as jnp
from jaxtyping import Array, Float

from renderers.rays import RayBundle


@dataclass
class RayGenerator:
    width: int
    height: int

    @abstractmethod
    def generate(self, *args, **kwargs) -> RayBundle:
        """Generate a ray bundle."""

    def __call__(self, *args, **kwargs) -> RayBundle:
        return self.generate(*args, **kwargs)


@dataclass
class Perspective(RayGenerator):
    focal: float

    def generate(self,
                 pixel_coordinates: Float[Array, "num_of_pixels 2"],
                 pose: Float[Array, "4 4"],
                 t_near: float, t_far: float) -> RayBundle:
        i = pixel_coordinates[..., 1] + 0.5
        j = pixel_coordinates[..., 0] + 0.5

        i = (i - self.width * 0.5) / self.focal
        j = -(j - self.height * 0.5) / self.focal
        k = -jnp.ones_like(i)

        dirs = jnp.stack([i, j, k], axis=-1)
        ray_dirs = jnp.einsum('ij,kj->ik', dirs, pose[:3, :3])

        ray_origins = jnp.broadcast_to(pose[:3, -1], jnp.shape(ray_dirs))

        return RayBundle(origins=ray_origins, directions=ray_dirs,
                         t_nears=t_near, t_fars=t_far)


@dataclass
class Parallel(RayGenerator):
    viewport_height: float
    """
    *Full* height of the viewport/projection plane in world coordinate, corresponding to
    2 * ParallelScale in VTK
    """

    def generate(self,
                 pixel_coordinates: Float[Array, "num_of_pixels 2"],
                 pose: Float[Array, "4 4"],
                 t_near: float,
                 t_far: float) -> RayBundle:
        i = pixel_coordinates[..., 1] + 0.5
        j = pixel_coordinates[..., 0] + 0.5

        i = (i / self.width - 0.5) * self.viewport_height
        j = -(j / self.height - 0.5) * self.viewport_height
        k = jnp.zeros_like(i)

        translates = jnp.stack([i, j, k], axis=-1)
        ray_origins = pose[:3, -1] + jnp.einsum('ij,kj->ik', translates, pose[:3, :3])

        ray_dirs = jnp.broadcast_to(-1 * pose[:3, 2], jnp.shape(ray_origins))

        return RayBundle(origins=ray_origins, directions=ray_dirs,
                         t_nears=t_near, t_fars=t_far)


# Note: camera.focal_point = camera.position + camera.distance * camera.direction
#  pose = [ +X | +Y | +Z | P], camera.direction = -Z
def generate_perspective_rays(pixel_coordinates, width, height, fov_deg, focal, pose):
    fov = jnp.radians(fov_deg)

    # shift pixel coordinates such that the center pixel is at (0, 0).
    pixel_coordinates = pixel_coordinates - jnp.array([int(width / 2), int(height / 2)])
    # change from (row, column) to (x, y)
    pixel_coordinates = pixel_coordinates[:, [1, 0]]

    dxdy = 2 * focal * jnp.tan(fov / 2.) / jnp.array([width, height])
    z = -jnp.ones(shape=(pixel_coordinates.shape[0], 1)) * focal
    pixel_coordinates = pixel_coordinates * dxdy
    directions = jnp.concatenate([pixel_coordinates, z], axis=1)

    # The columns of the upper left 3x3 submatrix of `pose` are the
    # unit vectors for camera x, y, z directions i.e. | +X | +Y | +Z |,
    # in the world coordinate system. They need to be elementwise
    # multiplied (scaled) by `direction`, aka (dx, dy, dz) of each
    # pixel, i.e.
    # | dx * +X | dy * +Y | dz * +Z|, then summed row-wise to get
    # | dx * +X + dy * +Y + dz * +Z|.
    # Note that each "column vector" is itself a 3-vector.
    # Note that ray_dirs are not normalized, it is the vector from camera position
    # to the pixel on the projection plane. We will get t = 1 for points on the
    # projection plane. The t value needs to be scaled by (normal of ray_dirs?)
    # to get the physical Z value in the world coordinate system.
    ray_dirs = jnp.sum(directions[..., jnp.newaxis, :] * pose[:3, :3], axis=-1)

    # The last column of `pose` is the camera position, i.e., the origin of rays.
    # It is the same for all the pixels (for perspective projection) thus we
    # broadcast it to a matrix with the shape of ray_dirs.
    ray_origins = jnp.broadcast_to(pose[:3, -1], jnp.shape(ray_dirs))

    return ray_origins, ray_dirs
