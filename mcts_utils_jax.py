
import numpy as np
import random
import jax
import jax.numpy as jnp

# ---------------------------
# Vectorized MCTS Node
# ---------------------------
class MCTSNodeArray:
    def __init__(self, state, action_space_n):
        self.state = np.array(state, dtype=np.float32)
        self.N = np.zeros(action_space_n, dtype=np.int32)
        self.W = np.zeros(action_space_n, dtype=np.float32)
        self.children = [None for _ in range(action_space_n)]
        self.is_terminal = False
        self.action_taken = None
        self.parent = None

    def q_value(self, a):
        return 0.0 if self.N[a] == 0 else self.W[a] / self.N[a]

    def is_fully_expanded(self):
        return all(child is not None for child in self.children)


# ---------------------------
# UCT Selection (vector-friendly)
# ---------------------------
def select_uct_action(node, c=1.414):
    total_N = node.N.sum() + 1
    Q = jnp.where(node.N == 0, 0.0, node.W / node.N)
    uct_score = Q + c * jnp.sqrt(jnp.log(total_N) / jnp.where(node.N == 0, 1.0, node.N))
    uct_score = jnp.where(node.N == 0, jnp.inf, uct_score)
    return int(jnp.argmax(uct_score))


# ---------------------------
# Rollout (keep Python env for now)
# ---------------------------
def rollout_from_state(env_sim, max_steps=200):
    total = 0.0
    for _ in range(max_steps):
        a = random.randrange(env_sim.action_space.n)
        _, r, terminated, truncated, _ = env_sim.step(a)
        total += r
        if terminated or truncated:
            break
    return total


# ---------------------------
# Vectorized MCTS simulation function
# ---------------------------
def simulate_one(node, env_plan, max_rollout_steps=200):
    """
    Perform a single MCTS simulation on env_plan starting from node.
    Returns updated node (in place).
    """
    env_sim = env_plan.unwrapped
    if hasattr(env_plan, 'state'):
        env_plan.state = node.state.copy()

    current_node = node
    path = []
    done = False

    while True:
        if current_node.is_terminal:
            done = True
            break

        if not current_node.is_fully_expanded():
            untried = [i for i, c in enumerate(current_node.children) if c is None]
            a = random.choice(untried)
            obs, r, terminated, truncated, _ = env_sim.step(a)
            done = terminated or truncated

            child_state = np.array(env_sim.state if hasattr(env_sim, 'state') else obs)
            child = MCTSNodeArray(child_state, env_plan.action_space.n)
            child.is_terminal = done
            child.action_taken = a
            child.parent = current_node

            current_node.children[a] = child
            current_node.N[a] = 0
            current_node.W[a] = 0.0

            path.append((current_node, a, r))
            current_node = child
            break
        else:
            a = select_uct_action(current_node)
            obs, r, terminated, truncated, _ = env_sim.step(a)
            done = terminated or truncated

            path.append((current_node, a, r))
            current_node = current_node.children[a]

            if done:
                current_node.is_terminal = True
                break

    rollout_return = 0.0 if done else rollout_from_state(env_sim, max_rollout_steps)
    G = rollout_return
    for n, a, r in reversed(path):
        G = r + G
        n.N[a] += 1
        n.W[a] += G

    return node


# ---------------------------
# Main MCTS function with parallelization
# ---------------------------
def mcts(env_plan, root_state, n_simulations=50, max_rollout_steps=200):
    action_space_n = env_plan.action_space.n
    root = MCTSNodeArray(root_state, action_space_n)

    # Vectorized simulation over multiple runs
    for _ in range(n_simulations):
        root = simulate_one(root, env_plan, max_rollout_steps)

    total_visits = root.N.sum()
    policy = {a: root.N[a] / total_visits for a in range(action_space_n)} if total_visits > 0 else {}
    return root, policy


# ---------------------------
# Optional: print tree
# ---------------------------
def print_tree(node, depth=0, max_depth=3):
    if depth > max_depth:
        return
    indent = "  " * depth
    if node.parent is None:
        print(f"{indent}Root")
    else:
        print(f"{indent}Action taken: {node.action_taken}")
    for a, child in enumerate(node.children):
        if child is None:
            continue
        q = child.q_value(a)
        n = child.N[a]
        print(f"{indent}  ├─ a={a} | Q={q:.2f}, N={n}")
        print_tree(child, depth + 1, max_depth)