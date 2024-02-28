import jax.numpy as jnp
import jax.random
import matplotlib.pyplot as plt
import optax
from flax.training.train_state import TrainState
from tqdm import tqdm

from models.nerfs import TinyNeRFModel, InstantNGP
from renderers.ray_gen import generate_rays
from renderers.volume import volume_rendering
from samplers.pixel import Dense


def create_train_step(key, model, optimizer, t_near, t_far):
    init_state = TrainState.create(apply_fn=model.apply,
                                   params=model.init(key, jnp.empty((1024, 3)), jnp.empty((1024, 3))),
                                   tx=optimizer)

    def loss_fn(params, inputs, target, key):
        ray_o, ray_d = inputs
        rgb, depth, _ = volume_rendering(model.bind(params), ray_o, ray_d, key,
                                         t_near=t_near, t_far=t_far)
        return jnp.mean(optax.l2_loss(rgb, target.reshape(-1, 3)))

    @jax.jit
    def train_step(state, inputs, target, key):
        loss_val, grads = jax.value_and_grad(loss_fn)(state.params, inputs, target, key)
        new_state = state.apply_gradients(grads=grads)
        return new_state, loss_val

    return train_step, init_state


if __name__ == "__main__":
    data = jnp.load("../data/tangle_tiny.npz")
    # data = jnp.load("../data/tiny_nerf_data.npz")

    images = data["images"][..., :3]

    height, width = images.shape[1], images.shape[2]

    poses = data["poses"].astype(jnp.float32)
    focal = data["focal"]

    t_near = 2.0
    t_far = 8.0

    key = jax.random.PRNGKey(12345)
    # model = TinyNeRFModel()
    model = InstantNGP()

    # it seems that the learning rate is sensitive to model, for ReLu, it is 1e-3
    # for Siren, it is 1.e-4
    schedule_fn = optax.exponential_decay(init_value=1e-3, transition_begin=600,
                                          transition_steps=200, decay_rate=0.5)
    optimizer = optax.adam(learning_rate=schedule_fn)

    train_step, state = create_train_step(key, model, optimizer, t_near, t_far)

    pbar = tqdm(range(5000))

    pixel_sampler = Dense(width=width, height=height)
    pixel_coordinates = pixel_sampler()

    for i in pbar:
        key, subkey = jax.random.split(key)
        image_idx = jax.random.randint(subkey, (1,), minval=0, maxval=images.shape[0])[0]

        image = images[image_idx]
        pose = poses[image_idx]

        ray_origins, ray_directions = generate_rays(pixel_coordinates,
                                                    width=width,
                                                    height=height,
                                                    focal=focal,
                                                    pose=pose)
        target = image[..., :3]

        key, subkey = jax.random.split(key)
        state, loss = train_step(state, (ray_origins, ray_directions), target, subkey)
        pbar.set_description("Loss %f" % loss)

        if i % 100 == 0:
            ray_origins, ray_directions = generate_rays(pixel_coordinates,
                                                        width=width,
                                                        height=height,
                                                        focal=focal,
                                                        pose=poses[-1])

            key, _ = jax.random.split(key)
            image_recon, depth_recon, _ = volume_rendering(model.bind(state.params), ray_origins,
                                                           ray_directions, key,
                                                           t_near, t_far)
            plt.imshow(image_recon.reshape((100, 100, 3)))
            plt.savefig(str(i).zfill(6) + "png")
            plt.close()
