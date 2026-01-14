import jax
import jax.numpy as jnp
import jax.random as random
from config import NUM_ACTIONS, MAX_NODES, UCT_C
from mcts.uct import select_uct

def mcts_step(rollout_fn):
    """
    Fully JAX-compatible MCTS step with:
    - batched simulations
    - deterministic preallocated tree expansion
    - JIT-friendly, no dynamic Python operations
    """
    def _step(carry, _):
        tree, sim_node_idx, state_batch, rng_key = carry
        batch_size = sim_node_idx.shape[0]

        # -------------------------
        # Selection
        # -------------------------
        rng_key, subkey = random.split(rng_key)
        actions = select_uct(tree["N"], tree["W"], sim_node_idx, subkey)

        # -------------------------
        # Expansion: deterministic preallocated children
        # -------------------------
        leaf_mask = tree["is_leaf"][sim_node_idx]  # [BATCH_SIZE]

        # Precompute new child indices for each simulation (deterministic)
        child_offset = jnp.arange(NUM_ACTIONS, dtype=jnp.int32)  # [0,1,...]
        leaf_mask = tree["is_leaf"][sim_node_idx]  # [BATCH_SIZE]

        # Allocate unique new nodes for each leaf
        start_id = tree["next_node_id"]
        new_children = jnp.arange(NUM_ACTIONS) + start_id


        # Update the tree
        tree["children"] = tree["children"].at[sim_node_idx].set(new_children)
        tree["is_leaf"] = tree["is_leaf"].at[new_children].set(True)

        # Increment next_node_id
        tree["next_node_id"] = start_id + NUM_ACTIONS


        # -------------------------
        # Rollout
        # -------------------------
        rng_key, subkey = random.split(rng_key)
        rewards = rollout_fn(state_batch, subkey)

        # Tiny noise to break ties
        rng_key, subkey = random.split(rng_key)
        rewards += 1e-3 * random.uniform(subkey, rewards.shape)

        # -------------------------
        # Backpropagation
        # -------------------------
        tree["N"] = tree["N"].at[sim_node_idx, actions].add(1)
        tree["W"] = tree["W"].at[sim_node_idx, actions].add(rewards)

        # -------------------------
        # Move simulation to child
        # -------------------------
        # sim_node_idx = tree["children"][sim_node_idx, actions]

        return (tree, sim_node_idx, state_batch, rng_key), None

    return _step