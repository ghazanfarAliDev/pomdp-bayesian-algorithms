import jax
import jax.random as random
from gymnax import make
from config import ENV_NAME, BATCH_SIZE

def make_env():
    env, env_params = make(ENV_NAME)
    return env, env_params

def reset_env_batch(env, env_params, rng_key):
    keys = random.split(rng_key, BATCH_SIZE)
    obs, state = jax.vmap(lambda k: env.reset(k, env_params))(keys)
    return obs, state
