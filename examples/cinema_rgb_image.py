import csv

import h5py
import jax
import jax.numpy as jnp
import numpy
import optax
from flax.training.train_state import TrainState
from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np
import pyvista as pv
from synema.models.cinema import CinemaRGBAImage
from synema.renderers.ray_gen import Parallel
from synema.renderers.rays import RayBundle
from synema.renderers.volume import Hierarchical
from synema.samplers.pixel import Dense, UniformRandom


def readCinemaDatabase():
    with open('/home/ollie/PycharmProjects/imdb_nerf/examples/cube_cinema.cdb/data.csv', 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')

        viewport_heights = []
        poses = []
        images = []
    
        for row in reader:
            h5file = h5py.File('/home/ollie/PycharmProjects/imdb_nerf/examples/cube_cinema.cdb/' + row['FILE'], 'r')
            meta = h5file.get('meta')

            camera_height = numpy.array(meta['CameraHeight'])  # not used by color image exporter
            viewport_heights.append(camera_height)

            camera_dir = numpy.array(meta['CameraDir'])
            camera_pos = numpy.array(meta['CameraPos'])
            camera_near_far = numpy.array(meta['CameraNearFar'])  # not used by color image exporter
            camera_up = numpy.array(meta['CameraUp'])

            # construct camera orientation matrix
            camera_w = -camera_dir / numpy.linalg.norm(camera_dir)
            camera_u = numpy.cross(camera_up, camera_w)
            camera_u = camera_u / numpy.linalg.norm(camera_u)
            camera_v = numpy.cross(camera_w, camera_u)
            camera_v = camera_v / numpy.linalg.norm(camera_v)

            # normalize the bbox to [-0.5, 0.5]^3 to prevent vanishing gradient.
            camera_pos_normalized = 0.5 * camera_w

            pose = numpy.zeros((4, 4))
            pose[:3, 0] = camera_u
            pose[:3, 1] = camera_v
            pose[:3, 2] = camera_w
            pose[:3, 3] = camera_pos_normalized
            pose[3, 3] = 1

            poses.append(pose)

            channels = h5file.get('channels')
            image = numpy.array(channels['rgba'], dtype=numpy.float32) / 255.
            images.append(image)

        viewport_heights = numpy.stack(viewport_heights, axis=0)
        poses = numpy.stack(poses, axis=0)
        images = numpy.stack(images, axis=0)

        return viewport_heights, poses, images


def create_train_steps(key, model, optimizer):
    init_state = TrainState.create(apply_fn=model.apply,
                                   params=model.init(key, jnp.empty((1024, 3)),
                                                     jnp.empty((1024, 3))),
                                   tx=optimizer)
    train_renderer = Hierarchical()

    def loss_fn(params, ray_bundle: RayBundle, targets, key: jax.random.PRNGKey):
        _, rgb, alpha, depth = train_renderer(coarse_field=model.bind(params),
                                              fine_field=model.bind(params),
                                              ray_bundle=ray_bundle,
                                              rng_key=key).values()
        return jnp.mean(optax.l2_loss(rgb, targets['rgb']))
        # return (jnp.mean(optax.l2_loss(scalar, targets['scalar'])) +
        #         1.e-3 * jnp.mean(jnp.abs(depth - jnp.nan_to_num(targets['depth']))))

    @jax.jit
    def train_step(state, ray_bundle: RayBundle, targets, key: jax.random.PRNGKey):
        loss_val, grads = jax.value_and_grad(loss_fn)(state.params, ray_bundle, targets, key)
        new_state = state.apply_gradients(grads=grads)
        return new_state, loss_val

    return train_step, init_state


if __name__ == '__main__':
    viewport_heights, poses, images = readCinemaDatabase()
    height, width = images.shape[1], images.shape[2]

    plt.imshow(images[-1])
    plt.savefig("rgb_gt")
    plt.close()

    t_near = 0.
    t_far = 1.

    key = jax.random.PRNGKey(0)
    model = CinemaRGBAImage()

    schedule_fn = optax.exponential_decay(init_value=1e-3, transition_begin=600,
                                          transition_steps=200, decay_rate=0.5)
    optimizer = optax.adam(learning_rate=schedule_fn)

    train_step, state = create_train_steps(key, model, optimizer)

    pixel_sampler = UniformRandom(width=width,
                                  height=height,
                                  n_samples=4096)

    ray_generator = Parallel(width=width, height=height, viewport_height=viewport_heights[0])
    renderer = Hierarchical()

    pbar = tqdm(range(1000))
    for i in pbar:
        key, subkey = jax.random.split(key)
        image_idx = jax.random.randint(subkey, shape=(1,), minval=1, maxval=images.shape[0])[0]

        pose = poses[image_idx]
        image = images[image_idx]

        key, subkey = jax.random.split(key)
        pixel_coordinates = pixel_sampler(rng=subkey)

        ray_bundle = ray_generator(pixel_coordinates, pose, t_near, t_far)
        targets = {
            'rgb': image[pixel_coordinates[:, 0].astype(int), pixel_coordinates[:, 1].astype(int), :3],
            'alpha': image[pixel_coordinates[:, 0].astype(int), pixel_coordinates[:, 1].astype(int), 3],
        }

        key, subkey = jax.random.split(key)
        state, loss = train_step(state, ray_bundle, targets, subkey)
        pbar.set_description("Loss: {:.8f}".format(loss))

        if i % 100 == 0:
            pixel_coordinates_infer = Dense(width=width, height=height)()
            ray_bundle = Parallel(width, height, viewport_heights[0])(pixel_coordinates_infer, poses[-1], t_near, t_far)

            key, _ = jax.random.split(key)
            _, image_recon, alpha_recon, depth_recon = renderer(model.bind(state.params),
                                                                model.bind(state.params),
                                                                ray_bundle,
                                                                key).values()

            plt.imshow(image_recon.reshape((width, height, 3)))
            plt.savefig(str(i).zfill(6) + "rgb")
            plt.close()
            plt.imshow(depth_recon.reshape((width, height, 1)))
            plt.colorbar()
            plt.savefig(str(i).zfill(6) + "depth")
            plt.close()

    # generate the vti
    # Recreate 100^3 grid in [-0.5, 0.5]^3
    grid_res = 100
    x = np.linspace(-1, 1, grid_res+1)
    y = np.linspace(-1, 1, grid_res+1)
    z = np.linspace(-1, 1, grid_res+1)
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    points = np.stack([zz.ravel(), yy.ravel(), xx.ravel()], axis=-1)
    print(points.shape)

    grid_res = 100
    spacing = (2.0 / grid_res, 2.0 / grid_res, 2.0 / grid_res)
    origin = (-1, -1, -1)

    # Create UniformGrid
    grid = pv.ImageData()
    grid.dimensions = (grid_res+1, grid_res+1, grid_res+1)
    grid.origin = origin
    grid.spacing = spacing


    field_fun = model.bind(state.params)
    array_of_rgb, array_of_density = field_fun(points, points)

    grid["density"] = array_of_density
    grid["rgb"] = array_of_rgb

    grid.save("cube_density_reconst.vti")
