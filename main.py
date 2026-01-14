import time
import jax
import jax.numpy as jnp
import jax.random as random
from copy import deepcopy

from config import RNG_SEED, BATCH_SIZE, MAX_SIMULATIONS
from envs.cartpole_env import make_env, reset_env_batch
from mcts.tree import init_tree
from mcts.rollout import make_rollout
from mcts.mcts_step import mcts_step
from visualize_tree import visualize_tree

def main():
    # -------------------------
    # RNG
    # -------------------------
    rng_key = random.PRNGKey(RNG_SEED)

    # -------------------------
    # Environment
    # -------------------------
    env, env_params = make_env()
    _, state_batch = reset_env_batch(env, env_params, rng_key)

    # -------------------------
    # MCTS components
    # -------------------------
    rollout_fn = make_rollout(env, env_params)
    step_fn = mcts_step(rollout_fn)

    # -------------------------
    # Initial tree + simulation state
    # -------------------------
    tree = init_tree()
    sim_node_idx = jnp.zeros((BATCH_SIZE,), dtype=jnp.int32)
    carry = (tree, sim_node_idx, state_batch, rng_key)

    # -------------------------
    # Warm-up (JIT compile)
    # -------------------------
    print("Running JIT warm-up...")
    carry, _ = jax.lax.scan(step_fn, carry, None, length=10)
    jax.block_until_ready(carry)

    # -------------------------
    # Timed MCTS run (per “step” style)
    # -------------------------
    print(f"Running {MAX_SIMULATIONS} MCTS simulations...")
    start = time.time()

    for sim in range(MAX_SIMULATIONS):
        # Use deepcopy for debugging/inspection if needed
        carry_copy = deepcopy(carry)

        # Perform a single MCTS step
        carry, _ = step_fn(carry_copy, None)
        jax.block_until_ready(carry)

        if sim % 100 == 0:  # print progress every 100 sims
            tree_snapshot, _, _, _ = carry
            root_policy = tree_snapshot["N"][0]
            root_policy = root_policy / (root_policy.sum() + 1e-8)
            print(f"Simulation {sim}: root policy = {root_policy}")

    end = time.time()

    # -------------------------
    # Results
    # -------------------------
    tree, _, _, _ = carry
    root_visits = tree["N"][0]
    policy = root_visits / (root_visits.sum() + 1e-8)

    print("Final root policy:", policy)
    print(f"Total time: {end - start:.4f}s")
    print(f"Time per simulation: {(end - start) / MAX_SIMULATIONS:.6f}s")

    # -------------------------
    # Tree Visualization
    # -------------------------
    print("Visualizing MCTS tree...")
    visualize_tree(tree, max_nodes=50)


if __name__ == "__main__":
    main()