import jax
import jax.numpy as jnp
from config import UCT_C

@jax.jit
def select_uct(N, W, node_idx):
    node_N = N[node_idx]
    node_W = W[node_idx]
    total_N = node_N.sum(axis=1, keepdims=True) + 1

    Q = jnp.where(node_N > 0, node_W / node_N, 0.0)
    U = UCT_C * jnp.sqrt(jnp.log(total_N) / (node_N + 1e-8))
    return jnp.argmax(Q + U, axis=1)
