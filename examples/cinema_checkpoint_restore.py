import jax
import jax.numpy as jnp
import jax.random
import optax
import orbax.checkpoint
from flax.training.train_state import TrainState
from tqdm import tqdm

from synema.models.cinema import CinemaScalarImage
from synema.renderers.ray_gen import Parallel
from synema.renderers.rays import RayBundle
from synema.renderers.volume import Hierarchical, DepthGuidedTrain
from synema.samplers.pixel import Dense


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


if __name__ == '__main__':
    ckpt_dir = '/home/ollie/PycharmProjects/synema/examples/checkpoints/cinema'

    key = jax.random.PRNGKey(0)
    model = CinemaScalarImage()

    schedule_fn = optax.exponential_decay(init_value=1e-3, transition_begin=600,
                                          transition_steps=200, decay_rate=0.5)
    optimizer = optax.adam(learning_rate=schedule_fn)

    train_step, state = create_train_steps(key, model, optimizer)
    target = {'model': state,
              'poses': jnp.zeros((0, 0, 0)),
              'depths': jnp.zeros((0, 0, 0)),
              'scalars': jnp.zeros((0, 0, 0))}

    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    raw_restored = checkpointer.restore(ckpt_dir, item=target)

    state = raw_restored['model']
    poses = raw_restored['poses']
    depths = raw_restored['depths']
    scalars = raw_restored['scalars']

    width, height = scalars.shape[1], scalars.shape[0]
    t_near = 0.
    t_far = 1.

    pixel_sampler = Dense(width=width, height=height)
    pixel_coordinates = pixel_sampler()

    ray_generator = Parallel(width=width, height=height, viewport_height=t_far)
    renderer = Hierarchical()

    pbar = tqdm(range(1000))
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
