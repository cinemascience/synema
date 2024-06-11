import csv

import cv2
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
from renderers.volume import Simple
from samplers.pixel import Dense


def readCinemaDatabase():
    with open('../data/cinema.cdb/data.csv', 'r', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader, None)  # skip header

        poses = []
        depths = []
        scalars = []

        for row in reader:
            h2file = h5py.File('../data/cinema.cdb/' + row[2], 'r')
            meta = h2file.get('meta')

            camera_focal = numpy.array(meta['CameraHeight'])
            camera_dir = numpy.array(meta['CameraDir'])
            camera_pos = numpy.array(meta['CameraPos'])
            camera_near_far = numpy.array(meta['CameraNearFar'])
            camera_up = numpy.array(meta['CameraUp'])
            # print(camera_near_far)

            # construct camera orientation matrix
            camera_w = -camera_dir
            camera_u = numpy.cross(camera_up, camera_w)
            camera_v = numpy.cross(camera_w, camera_u)
            # normalize basis vectors
            camera_u = camera_u / numpy.linalg.norm(camera_u)
            camera_v = camera_v / numpy.linalg.norm(camera_v)
            camera_w = camera_w / numpy.linalg.norm(camera_w)

            pose = numpy.zeros((4, 4))
            pose[:3, 0] = camera_u
            pose[:3, 1] = camera_v
            pose[:3, 2] = camera_w
            pose[:3, 3] = camera_pos
            pose[3, 3] = 1

            poses.append(pose)

            channels = h2file.get('channels')

            depth = numpy.array(channels['Depth'])
            depth = cv2.resize(depth, (128, 128))
            depths.append(depth)

            elevation = numpy.array(channels['Elevation'])
            elevation = cv2.resize(elevation, dsize=(128, 128))
            scalars.append(elevation)

        poses = numpy.stack(poses, axis=0)
        depths = numpy.stack(depths, axis=0)
        scalars = numpy.stack(scalars, axis=0)

        return poses, depths, scalars


def create_train_steps(key, model, optimizer):
    init_state = TrainState.create(apply_fn=model.apply, params=model.init(key, jnp.empty((1024, 3))), tx=optimizer)
    train_renderer = Simple()

    def loss_fn(params, ray_bundle: RayBundle, targets, key: jax.random.PRNGKey):
        scalar, alpha, depth = train_renderer(model.bind(params), ray_bundle, key)
        return jnp.mean(optax.l2_loss(scalar, targets['scalar'].reshape(-1, 1)))
        # return (jnp.mean(optax.l2_loss(scalar, targets['scalar'].reshape(-1, 1))) +
        #         1.e-3 * jnp.mean(optax.l2_loss(depth, targets['depth'].reshape(-1, 1))))

    @jax.jit
    def train_step(state, ray_bundle: RayBundle, targets, key: jax.random.PRNGKey):
        loss_val, grads = jax.value_and_grad(loss_fn)(state.params, ray_bundle, targets, key)
        new_state = state.apply_gradients(grads=grads)
        return new_state, loss_val

    return train_step, init_state


if __name__ == "__main__":
    poses, depths, scalars = readCinemaDatabase()
    scalars = jnp.nan_to_num(scalars)

    t_near = 0.
    t_far = 257.81808

    key = jax.random.PRNGKey(0)
    model = CinemaScalarImage()

    schedule_fn = optax.exponential_decay(init_value=5e-4, end_value=5e-5, transition_begin=500, transition_steps=200,
                                          decay_rate=0.9)
    optimizer = optax.adam(learning_rate=schedule_fn)

    train_step, state = create_train_steps(key, model, optimizer)

    pixel_sampler = Dense(width=128, height=128)
    pixel_coordinates = pixel_sampler()
    ray_generator = Parallel(128, 128, t_far)
    renderer = Simple()

    plt.imshow(scalars[0])
    plt.colorbar()
    plt.savefig('scalar_gt.png')

    pbar = tqdm(range(5000))

    for i in pbar:
        key, subkey = jax.random.split(key)
        image_idx = jax.random.randint(subkey, shape=(1,), minval=0, maxval=scalars.shape[0])[0]

        pose = poses[image_idx]
        depth = depths[image_idx]
        scalar = scalars[image_idx]

        ray_bundle = ray_generator(pixel_coordinates, pose, t_near, t_far)
        targets = {
            'scalar': scalar.reshape((-1, 1)),
            'depth': depth.reshape((-1, 1))}

        state, loss = train_step(state, ray_bundle, targets, subkey)
        pbar.set_description("Loss %f" % loss)

        if i % 100 == 0:
            ray_bundle = ray_generator(pixel_coordinates, poses[0], t_near, t_far)
            key, subkey = jax.random.split(key)
            scalar_recon, _, _ = renderer(model.bind(state.params), ray_bundle, subkey)

            plt.imshow(scalar_recon.reshape((128, 128)))
            plt.colorbar()
            plt.savefig(str(i).zfill(6) + 'scalar_recon.png')
            plt.close()
