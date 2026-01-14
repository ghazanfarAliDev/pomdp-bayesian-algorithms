import jax.numpy as jnp
from config import MAX_NODES, NUM_ACTIONS

def init_tree():
    return {
        "N": jnp.zeros((MAX_NODES, NUM_ACTIONS)),
        "W": jnp.zeros((MAX_NODES, NUM_ACTIONS)),
        "children": -jnp.ones((MAX_NODES, NUM_ACTIONS), dtype=int),
        "is_leaf": jnp.ones((MAX_NODES,), dtype=bool),
        "next_node_id": 1,
    }
