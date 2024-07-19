import csv

import flax.linen as nn
import h5py
import jax
import jax.numpy as jnp
import numpy
import optax
from flax.training.train_state import TrainState
from matplotlib import pyplot as plt
from tqdm import tqdm

from models.cinema import CinemaScalarImage
from renderers.ray_gen import Parallel
from renderers.rays import RayBundle
from renderers.volume import DepthGuidedTrain, Hierarchical
from samplers.pixel import Dense


def readCinemaDatabase():
    with open('../data/dragon.cdb/data.csv', 'r', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader, None)  # skip header

        poses = []
        depths = []
        scalars = []

        for row in reader:
            h2file = h5py.File('../data/dragon.cdb/' + row[2], 'r')
            meta = h2file.get('meta')

            camera_focal = numpy.array(meta['CameraHeight'])
            camera_dir = numpy.array(meta['CameraDir'])
            camera_pos = numpy.array(meta['CameraPos'])
            camera_near_far = numpy.array(meta['CameraNearFar'])
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

            channels = h2file.get('channels')

            depth = numpy.array(channels['Depth'])
            depths.append(depth)

            elevation = numpy.array(channels['Elevation'])
            scalars.append(elevation)

        poses = numpy.stack(poses, axis=0)
        depths = numpy.stack(depths, axis=0)
        scalars = numpy.stack(scalars, axis=0)

        return poses, depths, scalars


def create_train_steps(key, model, optimizer):
    init_state = TrainState.create(apply_fn=model.apply, params=model.init(key, jnp.empty((1024, 3))), tx=optimizer)
    train_renderer = DepthGuidedTrain()

    def loss_fn(params, ray_bundle: RayBundle, targets, key: jax.random.PRNGKey):
        scalar, alpha, depth = train_renderer(field_fn=model.bind(params),
                                              ray_bundle=ray_bundle,
                                              rng_key=key,
                                              depth_gt=targets['depth']).values()
        return (jnp.mean(optax.l2_loss(scalar, targets['scalar'])) +
                1.e-3 * jnp.mean(jnp.abs(depth - jnp.nan_to_num(targets['depth']))))

    @jax.jit
    def train_step(state, ray_bundle: RayBundle, targets, key: jax.random.PRNGKey):
        loss_val, grads = jax.value_and_grad(loss_fn)(state.params, ray_bundle, targets, key)
        new_state = state.apply_gradients(grads=grads)
        return new_state, loss_val

    return train_step, init_state


if __name__ == "__main__":
    poses, depths, scalars = readCinemaDatabase()
    height, width = scalars.shape[1], scalars.shape[2]
    scalars = jnp.nan_to_num(scalars)

    t_near = 0.
    t_far = 1.

    depths = jnp.where(depths == 1.0, jnp.nan, depths)

    plt.imshow(scalars[0])
    plt.colorbar()
    plt.savefig('scalar_gt.png')
    plt.close()

    plt.imshow(jnp.nan_to_num(depths[0]))
    plt.colorbar()
    plt.savefig('depth_gt.png')
    plt.close()

    key = jax.random.PRNGKey(0)
    model = CinemaScalarImage()

    schedule_fn = optax.exponential_decay(init_value=1e-3, transition_begin=600,
                                          transition_steps=200, decay_rate=0.5)
    optimizer = optax.adam(learning_rate=schedule_fn)

    train_step, state = create_train_steps(key, model, optimizer)

    pixel_sampler = Dense(width=width, height=height)
    pixel_coordinates = pixel_sampler()

    ray_generator = Parallel(width=width, height=height, viewport_height=t_far)
    renderer = Hierarchical()

    pbar = tqdm(range(5000))

    for i in pbar:
        key, subkey = jax.random.split(key)
        image_idx = jax.random.randint(subkey, shape=(1,), minval=1, maxval=scalars.shape[0])[0]

        pose = poses[image_idx]
        depth = depths[image_idx]
        scalar = scalars[image_idx]

        ray_bundle = ray_generator(pixel_coordinates, pose, t_near, t_far)
        targets = {
            'scalar': scalar.reshape((-1, 1)),
            'depth': depth.reshape((-1, 1))}

        key, subkey = jax.random.split(key)
        state, loss = train_step(state, ray_bundle, targets, subkey)
        pbar.set_description("Loss %f" % loss)

        if i % 100 == 0:
            ray_bundle = ray_generator(pixel_coordinates, poses[0], t_near, t_far)

            key, subkey = jax.random.split(key)
            _, scalar_recon, _, depth_recon = renderer(model.bind(state.params), model.bind(state.params),
                                                       ray_bundle, subkey).values()

            plt.imshow(scalar_recon.reshape((width, height)))
            plt.colorbar()
            plt.savefig(str(i).zfill(6) + 'scalar_recon.png')
            plt.close()

            plt.imshow(depth_recon.reshape((width, height, 1)))
            plt.colorbar()
            plt.savefig(str(i).zfill(6) + "depth")
            plt.close()

            plt.imshow(jnp.abs(depth_recon.reshape((width, height)) - jnp.nan_to_num(depths[0])))
            plt.colorbar()
            plt.savefig(str(i).zfill(6) + "depth_diff")
            plt.close()
