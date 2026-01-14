import jax
import jax.numpy as jnp
import jax.random as random
from config import UCT_C

@jax.jit
def select_uct(N, W, node_idx, rng_key):
    """
    Select actions using UCT with tie-breaking.
    """
    node_N = N[node_idx]  # [BATCH_SIZE, NUM_ACTIONS]
    node_W = W[node_idx]
    total_N = node_N.sum(axis=1, keepdims=True) + 1e-8

    Q = jnp.where(node_N > 0, node_W / node_N, 0.0)
    U = UCT_C * jnp.sqrt(jnp.log(total_N) / (node_N + 1e-8))

    scores = Q + U
    scores = scores + 1e-6 * random.uniform(rng_key, scores.shape)
    return jnp.argmax(scores, axis=1)