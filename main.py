import time
import jax
import jax.numpy as jnp
import jax.random as random

from config import (
    RNG_SEED,
    BATCH_SIZE,
    MAX_SIMULATIONS,
)

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
    # Initial tree + sim state
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
    # Timed MCTS run
    # -------------------------
    print(f"Running {MAX_SIMULATIONS} MCTS simulations...")
    start = time.time()

    carry, _ = jax.lax.scan(step_fn, carry, None, length=MAX_SIMULATIONS)
    jax.block_until_ready(carry)

    end = time.time()

    # -------------------------
    # Results
    # -------------------------
    tree, _, _, _ = carry
    root_visits = tree["N"][0]
    policy = root_visits / (root_visits.sum() + 1e-8)

    print("Root policy:", policy)
    print(f"Total time: {end - start:.4f}s")
    print(f"Time per simulation: {(end - start) / MAX_SIMULATIONS:.6f}s")

    # -------------------------
    # Tree Visualization
    # -------------------------
    print("Visualizing MCTS tree...")
    visualize_tree(tree, max_nodes=500)


if __name__ == "__main__":
    main()
