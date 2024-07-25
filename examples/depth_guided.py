import jax.numpy as jnp
import jax.random
import matplotlib.pyplot as plt
import optax
from flax.training.train_state import TrainState
from tqdm import tqdm

from synema.renderers.ray_gen import Perspective
from synema.renderers.rays import RayBundle
from synema.renderers.volume import DepthGuidedInfer, DepthGuidedTrain
from synema.samplers.pixel import Dense


def create_train_step(key, model, optimizer):
    init_state = TrainState.create(apply_fn=model.apply,
                                   params=model.init(key, jnp.empty((1024, 3)), jnp.empty((1024, 3))),
                                   tx=optimizer)
    train_renderer = DepthGuidedTrain()

    def loss_fn(params, ray_bundle: RayBundle, targets, key: jax.random.PRNGKey):
        rgb, alpha, depth = train_renderer(field_fn=model.bind(params),
                                           ray_bundle=ray_bundle,
                                           rng_key=key,
                                           depth_gt=-targets['depth'].reshape((-1, 1))).values()
        return (jnp.mean(optax.l2_loss(rgb, targets['rgb'].reshape(-1, 3))) +
                0.05 * jnp.mean(optax.l2_loss(alpha, targets['alpha'].reshape(-1, ))) +
                1.e-5 * jnp.mean(jnp.abs(depth + jnp.nan_to_num(targets['depth'].reshape((-1,))))))

    @jax.jit
    def train_step(state, ray_bundle: RayBundle, targets, key: jax.random.PRNGKey):
        loss_val, grads = jax.value_and_grad(loss_fn)(state.params, ray_bundle, targets, key)
        new_state = state.apply_gradients(grads=grads)
        return new_state, loss_val

    return train_step, init_state


if __name__ == "__main__":
    data = jnp.load("../data/tangle_tiny.npz")

    images = data["images"]
    height, width = images.shape[1], images.shape[2]

    depths = data["depths"].astype(jnp.float32)

    plt.imshow(-jnp.nan_to_num(depths[-1]))
    plt.colorbar()
    plt.savefig("depth_gt")
    plt.close()

    plt.imshow(images[-1])
    plt.savefig("rgb_gt")
    plt.close()

    poses = data["poses"].astype(jnp.float32)
    focal = data["focal"].item()

    t_near = 2.0
    t_far = 8.0

    key = jax.random.PRNGKey(12345)
    # model = VeryTinyNeRFModel()
    model = SirenNeRFModel()

    # it seems that the learning rate is sensitive to model, for ReLu, it is 1e-3
    # for Siren, it is 1.e-4
    schedule_fn = optax.exponential_decay(init_value=1e-3, transition_begin=600,
                                          transition_steps=200, decay_rate=0.5)
    optimizer = optax.adam(learning_rate=schedule_fn)

    train_step, state = create_train_step(key, model, optimizer)

    pixel_sampler = Dense(width=width, height=height)
    pixel_coordinates = pixel_sampler()

    # pixel_sampler = UniformRandom(width=width,
    #                               height=height,
    #                               n_samples=1024)

    ray_generator = Perspective(width=width, height=height, focal=focal)

    infer_renderer = DepthGuidedInfer()

    pbar = tqdm(range(5000))
    for i in pbar:
        key, subkey = jax.random.split(key)
        image_idx = jax.random.randint(subkey, (1,), minval=0, maxval=images.shape[0] - 1)[0]

        image = images[image_idx]
        depth = depths[image_idx]
        pose = poses[image_idx]

        # key, subkey = jax.random.split(key)
        # pixel_coordinates = pixel_sampler(mask, key)
        # pixel_coordinates = pixel_sampler(key)

        ray_bundle = ray_generator(pixel_coordinates, pose, t_near, t_far)

        targets = {'rgb': image[..., :3], 'alpha': image[..., 3], 'depth': depth}
        # targets = {'rgb': image[pixel_coordinates[:, 0].astype(int), pixel_coordinates[:, 1].astype(int), :3],
        #            'depth': depth[pixel_coordinates[:, 0].astype(int), pixel_coordinates[:, 1].astype(int)]}

        key, subkey = jax.random.split(key)
        state, loss = train_step(state, ray_bundle, targets, subkey)
        pbar.set_description("Loss %f" % loss)

        if i % 100 == 0:
            # pixel_coordinates = Dense(width=width, height=height)()
            ray_bundle = ray_generator(pixel_coordinates, poses[-1], t_near, t_far)

            key, _ = jax.random.split(key)
            image_recon, alpha_recon, depth_recon = infer_renderer(model.bind(state.params),
                                                                   ray_bundle,
                                                                   key).values()

            plt.imshow(image_recon.reshape((100, 100, 3)))
            plt.savefig(str(i).zfill(6) + "rgb")
            plt.close()
            plt.imshow(depth_recon.reshape((100, 100, 1)))
            plt.colorbar()
            plt.savefig(str(i).zfill(6) + "depth")
            plt.close()
            plt.imshow(jnp.abs(depth_recon.reshape((100, 100)) + jnp.nan_to_num(depths[-1])))
            plt.colorbar()
            plt.savefig(str(i).zfill(6) + "depth_diff")
            plt.close()
            plt.imshow(optax.l2_loss(image_recon.reshape((100, 100, 3)) - image[..., :3, -1]))
            plt.colorbar()
            plt.savefig(str(i).zfill(6) + "rgb_diff")
            plt.close()
            plt.imshow(jnp.sum(alpha_recon.reshape((100, 100, -1)), axis=-1))
            plt.colorbar()
            plt.savefig(str(i).zfill(6) + "alpha")
            plt.close()
