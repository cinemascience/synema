import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
from matplotlib import pyplot as plt
from tqdm import tqdm

from synema.models.nerfs import NeRFModel
from synema.renderers.ray_gen import Perspective
from synema.renderers.rays import RayBundle
from synema.renderers.volume import Hierarchical
from synema.samplers.pixel import Dense, UniformRandom


def create_train_step(key, model_coarse, model_fine, optimizer):
    rng_coarse, rng_fine = jax.random.split(key, 2)
    init_states = TrainState.create(apply_fn=None,
                                    params={'params_coarse': (model_coarse.init(rng_coarse,
                                                                                jnp.empty((1024, 3)),
                                                                                jnp.empty((1024, 3)))),
                                            'params_fine': (model_fine.init(rng_fine,
                                                                            jnp.empty((1024, 3)),
                                                                            jnp.empty((1024, 3))))},
                                    tx=optimizer)

    train_renderer = Hierarchical()

    def loss_fn(params, ray_bundle: RayBundle, target, key):
        rgb_c, rgb_f, alpha, depth = train_renderer(model_coarse.bind(params['params_coarse']),
                                                    model_fine.bind(params['params_fine']),
                                                    ray_bundle,
                                                    key).values()
        return (jnp.mean(optax.l2_loss(rgb_c, target.reshape(-1, 3))) +
                jnp.mean(optax.l2_loss(rgb_f, target.reshape(-1, 3))))

    @jax.jit
    def train_step(state, inputs, target, key):
        loss_val, grads = jax.value_and_grad(loss_fn)(state.params, inputs, target, key)
        new_state = state.apply_gradients(grads=grads)
        return new_state, loss_val

    return train_step, init_states


if __name__ == '__main__':
    # data = jnp.load("../data/tiny_nerf_data.npz")
    data = jnp.load("../data/tangle_tiny.npz")

    images = data["images"][..., :3]
    height, width = images.shape[1], images.shape[2]
    alphas = data['images'][..., 3]
    depths = jnp.nan_to_num(data["depths"].astype(jnp.float32), nan=0.)

    poses = data["poses"].astype(jnp.float32)
    focal = data["focal"]

    t_near = 2.0
    t_far = 8.0

    key = jax.random.PRNGKey(54321)
    model_coarse = NeRFModel(num_hidden_layers=4, num_hidden_features=64)
    model_fine = NeRFModel(num_hidden_layers=4, num_hidden_features=64)

    schedule_fn = optax.exponential_decay(init_value=5e-4, end_value=5e-5,
                                          transition_begin=500, transition_steps=200,
                                          decay_rate=0.8)
    optimizer = optax.adam(learning_rate=schedule_fn)

    train_step, states = create_train_step(key, model_coarse, model_fine,
                                           optimizer)

    pbar = tqdm(range(5000))

    pixel_sampler = UniformRandom(width=width,
                                  height=height,
                                  n_samples=1024)
    ray_generator = Perspective(width=width, height=height, focal=focal)
    renderer = Hierarchical()

    for i in pbar:
        key, subkey = jax.random.split(key)
        image_idx = jax.random.randint(subkey, (1,), minval=0, maxval=100)[0]

        image = images[image_idx]
        pose = poses[image_idx]

        key, subkey = jax.random.split(key)
        pixel_coordinates = pixel_sampler(rng=subkey)

        ray_bundle = ray_generator(pixel_coordinates, pose, t_near, t_far)

        target = image[pixel_coordinates[:, 0].astype(int), pixel_coordinates[:, 1].astype(int), :3]

        key, subkey = jax.random.split(key)
        states, loss = train_step(states, ray_bundle, target, subkey)
        pbar.set_description("Loss %f" % loss)

        if i % 100 == 0:
            pixel_coordinates = Dense(width=width, height=height)()

            ray_bundle = ray_generator(pixel_coordinates, poses[-1], t_near, t_far)

            key, _ = jax.random.split(key)
            _, image_recon, alpha_recon, depth_recon = renderer(model_coarse.bind(states.params['params_coarse']),
                                                                model_fine.bind(states.params['params_fine']),
                                                                ray_bundle,
                                                                key).values()
            plt.imshow(image_recon.reshape((100, 100, 3)))
            plt.savefig(str(i).zfill(6) + "rgb")
            plt.close()
            plt.imshow(optax.l2_loss(image_recon.reshape((100, 100, 3)) - images[..., :3, -1]))
            plt.colorbar()
            plt.savefig(str(i).zfill(6) + "rgb_diff")
            plt.close()

            plt.imshow(depth_recon.reshape((100, 100, 1)))
            plt.colorbar()
            plt.savefig(str(i).zfill(6) + "depth")
            plt.close()
            plt.imshow(jnp.abs(depth_recon.reshape((100, 100)) + depths[-1]))
            plt.colorbar()
            plt.savefig(str(i).zfill(6) + "depth_diff")
            plt.close()

            plt.imshow(jnp.sum(alpha_recon.reshape((100, 100, -1)), axis=-1))
            plt.colorbar()
            plt.savefig(str(i).zfill(6) + "alpha")
            plt.close()
            plt.imshow(jnp.abs(alpha_recon.reshape((100, 100)) - alphas[-1]))
            plt.colorbar()
            plt.savefig(str(i).zfill(6) + "alpha_diff")
            plt.close()
