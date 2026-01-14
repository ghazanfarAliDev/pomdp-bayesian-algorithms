import jax
import jax.numpy as jnp
import jax.random as random
from config import NUM_ACTIONS, BATCH_SIZE, MAX_ROLLOUT_STEPS

def make_rollout(env, env_params):

    @jax.jit
    def rollout(state_batch, rng_key):
        done = jnp.zeros((BATCH_SIZE,), dtype=bool)
        total_reward = jnp.zeros((BATCH_SIZE,))
        keys = random.split(rng_key, BATCH_SIZE)

        def step_fn(carry, _):
            states, keys, done, total_reward = carry
            actions = jax.vmap(
                lambda k: random.randint(k, (), 0, NUM_ACTIONS)
            )(keys)

            def step_env(s, a, k):
                _, next_s, r, d, _ = env.step(k, s, a, env_params)
                return next_s, r, d

            next_states, rewards, dones = jax.vmap(step_env)(states, actions, keys)
            total_reward += rewards * (~done)
            done = done | dones
            return (next_states, keys, done, total_reward), None

        carry = (state_batch, keys, done, total_reward)
        carry, _ = jax.lax.scan(step_fn, carry, None, length=MAX_ROLLOUT_STEPS)
        return carry[-1]

    return rollout
