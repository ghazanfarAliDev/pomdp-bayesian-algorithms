import jax
import jax.numpy as jnp
from gymnax import make
import jax.random as random
import time

# -------------------------
# Environment
# -------------------------
env, env_params = make("CartPole-v1")
NUM_ACTIONS = 2
rng_key = random.PRNGKey(0)

# -------------------------
# MCTS params
# -------------------------
BATCH_SIZE = 16
MAX_NODES = 1024
MAX_SIMULATIONS = 50000
MAX_ROLLOUT_STEPS = 50
UCT_C = 1.414

# -------------------------
# Tree arrays
# -------------------------
def init_tree():
    return {
        "N": jnp.zeros((MAX_NODES, NUM_ACTIONS)),
        "W": jnp.zeros((MAX_NODES, NUM_ACTIONS)),
        "children": -jnp.ones((MAX_NODES, NUM_ACTIONS), dtype=int),
        "is_leaf": jnp.ones((MAX_NODES,), dtype=bool),
        "next_node_id": 1,
    }

# -------------------------
# Batched env reset
# -------------------------
keys = random.split(rng_key, BATCH_SIZE)
obs_batch, state_batch = jax.vmap(lambda k: env.reset(k, env_params))(keys)
sim_node_idx = jnp.zeros((BATCH_SIZE,), dtype=int)  # all start at root

# -------------------------
# Rollout
# -------------------------
@jax.jit
def rollout(state_batch, rng_key, max_steps=MAX_ROLLOUT_STEPS):
    batch_size = BATCH_SIZE
    done = jnp.zeros((batch_size,), dtype=bool)
    total_reward = jnp.zeros((batch_size,))
    keys = random.split(rng_key, batch_size)

    def step_fn(carry, _):
        states, keys, done, total_reward = carry
        actions = jax.vmap(lambda k: random.randint(k, (), 0, NUM_ACTIONS))(keys)

        def step_env(s, a, k):
            _, next_s, r, d, _ = env.step(k, s, a, env_params)
            return next_s, r, d

        next_states, rewards, dones = jax.vmap(step_env)(states, actions, keys)
        total_reward += rewards * (~done)
        done = done | dones
        return (next_states, keys, done, total_reward), None

    carry = (state_batch, keys, done, total_reward)
    carry, _ = jax.lax.scan(step_fn, carry, None, length=max_steps)
    _, _, _, total_reward = carry
    return total_reward

# -------------------------
# UCT selection
# -------------------------
@jax.jit
def select_uct(N, W, node_idx):
    node_N = N[node_idx]
    node_W = W[node_idx]
    total_N = node_N.sum(axis=1, keepdims=True) + 1
    Q = jnp.where(node_N > 0, node_W / node_N, 0.0)
    U = UCT_C * jnp.sqrt(jnp.log(total_N) / (node_N + 1e-8))
    return jnp.argmax(Q + U, axis=1)

# -------------------------
# MCTS step (JAX-compatible)
# -------------------------
def mcts_step(carry, _):
    tree, sim_node_idx, state_batch, rng_key = carry

    # Selection
    actions = select_uct(tree["N"], tree["W"], sim_node_idx)

    # Expansion mask
    leaf_mask = tree["is_leaf"][sim_node_idx]
    new_nodes = jnp.arange(MAX_NODES)
    tree["children"] = tree["children"].at[sim_node_idx].set(jnp.where(
        leaf_mask[:, None], new_nodes[:BATCH_SIZE, None], tree["children"][sim_node_idx]
    ))
    tree["is_leaf"] = tree["is_leaf"].at[new_nodes[:BATCH_SIZE]].set(True)

    # Rollout
    rng_key, subkey = random.split(rng_key)
    rollout_rewards = rollout(state_batch, subkey)

    # Backpropagation
    tree["N"] = tree["N"].at[sim_node_idx, actions].add(1)
    tree["W"] = tree["W"].at[sim_node_idx, actions].add(rollout_rewards)

    # Move simulations to children
    sim_node_idx = jnp.where(leaf_mask, new_nodes[:BATCH_SIZE], tree["children"][sim_node_idx, actions])

    return (tree, sim_node_idx, state_batch, rng_key), None

# -------------------------
# Run full MCTS with timing
# -------------------------
tree_init = init_tree()
carry = (tree_init, sim_node_idx, state_batch, rng_key)

# Warm-up run to trigger JIT compilation
print("Running warm-up for JIT compilation...")
start_warmup = time.time()
carry_warmup, _ = jax.lax.scan(mcts_step, carry, None, length=10)
jax.block_until_ready(carry_warmup)
print(f"Warm-up done in {time.time() - start_warmup:.4f} seconds\n")

# Actual timed run
print(f"Running MCTS for {MAX_SIMULATIONS} simulations...")
start_time = time.time()
carry_final, _ = jax.lax.scan(mcts_step, carry, None, length=MAX_SIMULATIONS)
jax.block_until_ready(carry_final)  # ensure all GPU computation is finished
end_time = time.time()

tree_final, sim_node_idx, _, _ = carry_final
root_visits = tree_final["N"][0]
policy = root_visits / root_visits.sum()

print("Root policy (action probabilities):", policy)
print(f"Total MCTS run took {end_time - start_time:.4f} seconds")
print(f"Average time per simulation: {(end_time - start_time)/MAX_SIMULATIONS:.6f} seconds")