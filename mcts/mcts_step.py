import jax
import jax.numpy as jnp
import jax.random as random

from config import BATCH_SIZE, MAX_NODES
from mcts.uct import select_uct


def mcts_step(rollout_fn):
    """
    Returns a JAX-compatible MCTS step function for lax.scan
    """

    def _step(carry, _):
        tree, sim_node_idx, state_batch, rng_key = carry

        # -------------------------
        # Selection
        # -------------------------
        actions = select_uct(tree["N"], tree["W"], sim_node_idx)

        # -------------------------
        # Expansion
        # -------------------------
        leaf_mask = tree["is_leaf"][sim_node_idx]

        # Allocate new nodes safely
        next_node_id = tree["next_node_id"]
        new_nodes = next_node_id + jnp.arange(BATCH_SIZE)
        tree["next_node_id"] += BATCH_SIZE

        # Update children and leaves
        tree["children"] = tree["children"].at[sim_node_idx].set(
            jnp.where(
                leaf_mask[:, None],
                new_nodes[:, None],
                tree["children"][sim_node_idx]
            )
        )
        tree["is_leaf"] = tree["is_leaf"].at[new_nodes].set(True)

        # -------------------------
        # Rollout
        # -------------------------
        rng_key, subkey = random.split(rng_key)
        rewards = rollout_fn(state_batch, subkey)

        # -------------------------
        # Backpropagation
        # -------------------------
        tree["N"] = tree["N"].at[sim_node_idx, actions].add(1)
        tree["W"] = tree["W"].at[sim_node_idx, actions].add(rewards)

        # -------------------------
        # Move simulation to child
        # -------------------------
        sim_node_idx = jnp.where(
            leaf_mask,
            new_nodes,
            tree["children"][sim_node_idx, actions]
        )

        return (tree, sim_node_idx, state_batch, rng_key), None

    return _step
