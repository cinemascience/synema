import jax.numpy as jnp
import jax.random
import matplotlib.pyplot as plt
import optax
from flax.training.train_state import TrainState
from tqdm import tqdm

from models.nerfs import TinyNeRFModel, InstantNGP, VeryTinyNeRFModel
from renderers.ray_gen import Perspective
from renderers.rays import RayBundle
from renderers.volume import Simple
from samplers.pixel import Dense


def create_train_step(key, model, optimizer):
    init_state = TrainState.create(apply_fn=model.apply,
                                   params=model.init(key, jnp.empty((1024, 3)), jnp.empty((1024, 3))),
                                   tx=optimizer)
    train_renderer = Simple()

    def loss_fn(params, ray_bundle: RayBundle, targets, key: jax.random.PRNGKey):
        rgb, alpha, depth = train_renderer(model.bind(params),
                                           ray_bundle,
                                           key)
        return (jnp.mean(optax.l2_loss(rgb, targets['rgb'].reshape(-1, 3))) +
                1.e-3 * jnp.mean(optax.l2_loss(alpha, targets['depth'].reshape(-1, ))))

    # @jax.jit
    def train_step(state, ray_bundle: RayBundle, target, key: jax.random.PRNGKey):
        loss_val, grads = jax.value_and_grad(loss_fn)(state.params, ray_bundle, target, key)
        new_state = state.apply_gradients(grads=grads)
        return new_state, loss_val

    return train_step, init_state


if __name__ == "__main__":
    data = jnp.load("../data/tangle_tiny.npz")
    # data = jnp.load("../data/tiny_nerf_data.npz")

    images = data["images"]
    height, width = images.shape[1], images.shape[2]

    depths = jnp.nan_to_num(data["depths"].astype(jnp.float32), nan=0.)
    plt.imshow(depths[-1])
    plt.colorbar()
    plt.savefig("depth_gt")
    plt.close()

    poses = data["poses"].astype(jnp.float32)
    focal = data["focal"].item()

    t_near = 2.0
    t_far = 8.0

    key = jax.random.PRNGKey(12345)
    model = VeryTinyNeRFModel()
    # model = InstantNGP()

    # it seems that the learning rate is sensitive to model, for ReLu, it is 1e-3
    # for Siren, it is 1.e-4
    schedule_fn = optax.exponential_decay(init_value=1e-3, transition_begin=600,
                                          transition_steps=200, decay_rate=0.5)
    optimizer = optax.adam(learning_rate=schedule_fn)

    train_step, state = create_train_step(key, model, optimizer)

    pbar = tqdm(range(5000))

    pixel_sampler = Dense(width=width, height=height)
    pixel_coordinates = pixel_sampler()
    ray_generator = Perspective(width=width, height=height, focal=focal)
    renderer = Simple()

    for i in pbar:
        key, subkey = jax.random.split(key)
        image_idx = jax.random.randint(subkey, (1,), minval=0, maxval=images.shape[0])[0]

        image = images[image_idx]
        depth = depths[image_idx]
        pose = poses[image_idx]

        ray_bundle = ray_generator(pixel_coordinates, pose, t_near, t_far)

        # targets = (image[..., :3], depth)
        targets = {"rgb": image[..., :3], "alpha": image[..., 3], "depth": depth}

        key, subkey = jax.random.split(key)
        state, loss = train_step(state, ray_bundle, targets, subkey)
        pbar.set_description("Loss %f" % loss)

        if i % 100 == 0:
            ray_bundle = ray_generator(pixel_coordinates, poses[-1], t_near, t_far)

            key, _ = jax.random.split(key)
            image_recon, alpha_recon, depth_recon = renderer(model.bind(state.params),
                                                             ray_bundle,
                                                             key)

            plt.imshow(image_recon.reshape((100, 100, 3)))
            plt.savefig(str(i).zfill(6) + "rgb")
            plt.close()
            plt.imshow(depth_recon.reshape((100, 100, 1)))
            plt.colorbar()
            plt.savefig(str(i).zfill(6) + "depth")
            plt.close()
            plt.imshow(jnp.abs(depth_recon.reshape((100, 100)) + depths[-1]))
            plt.colorbar()
            plt.savefig(str(i).zfill(6) + "depth_diff")
            plt.close()
            plt.imshow(optax.l2_loss(image_recon.reshape((100, 100, 3)) - images[..., :3, -1]))
            plt.colorbar()
            plt.savefig(str(i).zfill(6) + "rgb_diff")
            plt.close()
            plt.imshow(alpha_recon.reshape((100, 100)))
            plt.colorbar()
            plt.savefig(str(i).zfill(6) + "alpha")
            plt.close()
