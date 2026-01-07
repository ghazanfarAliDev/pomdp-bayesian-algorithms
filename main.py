import gymnasium as gym
import numpy as np
import random
import matplotlib.pyplot as plt

class MCTSNode:
    def __init__(self, state, parent=None, action_taken=None):
        self.state = np.array(state, dtype=np.float32)
        self.parent = parent
        self.action_taken = action_taken
        self.children = {}  # action -> child node
        self.visits = 0
        self.value = 0.0

    def is_leaf(self):
        return len(self.children) == 0

    def q_value(self):
        return self.value / self.visits if self.visits > 0 else 0

def rollout_from_state(base_state, first_action, max_steps=200):
    env_sim = gym.make("CartPole-v1")
    obs, info = env_sim.reset()
    env_sim.unwrapped.state = np.array(base_state, dtype=np.float32)

    total_reward = 0.0
    obs, reward, terminated, truncated, info = env_sim.step(first_action)
    total_reward += reward
    done = terminated or truncated
    steps = 1

    while not done and steps < max_steps:
        a = random.choice([0, 1])
        obs, reward, terminated, truncated, info = env_sim.step(a)
        total_reward += reward
        done = terminated or truncated
        steps += 1

    env_sim.close()
    return total_reward

# ---------------- MCTS Algorithm ----------------
def mcts(env, root_state, n_simulations=50, max_rollout_steps=200):
    root = MCTSNode(root_state)

    for _ in range(n_simulations):
        node = root
        path = [node]

        # --- Selection ---
        while not node.is_leaf() and node.children:
            total_visits = sum(c.visits for c in node.children.values()) + 1
            best_score = -np.inf
            best_child = None
            for a, child in node.children.items():
                if child.visits == 0:
                    score = np.inf
                else:
                    score = child.q_value() + np.sqrt(2 * np.log(total_visits) / child.visits)
                if score > best_score:
                    best_score = score
                    best_child = child
            node = best_child
            path.append(node)

        # --- Expansion ---
        if node.visits == 0:
            for action in [0, 1]:
                if action not in node.children:
                    node.children[action] = MCTSNode(state=node.state, parent=node, action_taken=action)

        # --- Simulation ---
        first_action = random.choice([0, 1])
        reward = rollout_from_state(node.state, first_action, max_steps=max_rollout_steps)

        # --- Backpropagation ---
        for n in path:
            n.visits += 1
            n.value += reward

    # Policy at root
    policy = {action: child.visits / root.visits for action, child in root.children.items()}
    return root, policy

# ---------------- Print Tree ----------------
def print_tree(node, depth=0, max_depth=3):
    if depth > max_depth:
        return
    indent = "  " * depth
    print(f"{indent}- Action:{node.action_taken} | Q={node.q_value():.2f}, V={node.visits}")
    for child in node.children.values():
        print_tree(child, depth + 1, max_depth)

# ---------------- Tree Visualization (Matplotlib) ----------------
def visualize_tree_matplotlib(root, max_depth=3):
    fig, ax = plt.subplots(figsize=(12, 8))

    positions = {}
    labels = {}
    colors = []

    def add_node(node, x, y, level=0):
        node_id = id(node)
        positions[node_id] = (x, y)
        labels[node_id] = f"Q:{node.q_value():.1f}\nV:{node.visits}"
        normalized = min(max(node.q_value() / 50, 0), 1)
        colors.append((1 - normalized, normalized, 0))  # Red -> Green

        # Draw edges to children
        if level < max_depth and node.children:
            n_children = len(node.children)
            dx = 1 / (n_children + 1)
            i = 1
            for child in node.children.values():
                child_x = x - 0.5 + i * dx
                child_y = y - 1
                ax.plot([x, child_x], [y, child_y], 'k-')
                # Add edge label (action)
                mid_x = (x + child_x) / 2
                mid_y = (y + child_y) / 2
                ax.text(mid_x, mid_y, str(child.action_taken), color='blue', fontsize=12)
                add_node(child, child_x, child_y, level + 1)
                i += 1

    add_node(root, 0, 0)

    # Draw nodes
    for node_id, (x, y) in positions.items():
        ax.scatter(x, y, s=2500, c=[colors[list(positions.keys()).index(node_id)]])
        ax.text(x, y, labels[node_id], ha='center', va='center', fontsize=10)

    ax.axis('off')
    plt.show()

if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    obs, info = env.reset()
    root_node, policy = mcts(env, env.unwrapped.state, n_simulations=50)

    print("Policy at root:", policy)
    print("\nMCTS Tree Values:")
    print_tree(root_node, max_depth=3)

    visualize_tree_matplotlib(root_node, max_depth=3)