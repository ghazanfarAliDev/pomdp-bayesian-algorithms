import gymnasium as gym
import numpy as np
import random
import matplotlib.pyplot as plt
import copy

class MCTSNode:
    """
    Stores statistics on edges (s,a):
      N[a] = visit count
      W[a] = total return
      Q(s,a) = W[a] / N[a]
    """
    def __init__(self, state, parent=None, action_taken=None):
        self.state = np.array(state, dtype=np.float32)
        self.parent = parent
        self.action_taken = action_taken

        self.children = {}   # action -> MCTSNode
        self.N = {}          # action -> visit count
        self.W = {}          # action -> total return
        self.is_terminal = False

    def q_value(self, a):
        if self.N.get(a, 0) == 0:
            return 0.0
        return self.W[a] / self.N[a]

    def is_fully_expanded(self, action_space_n):
        return len(self.children) == action_space_n


# UCT SELECTION
def select_uct_action(node, action_space_n, c=1.414):
    total_N = sum(node.N.values()) + 1
    best_a, best_score = None, -np.inf

    for a in range(action_space_n):
        n_sa = node.N.get(a, 0)
        if n_sa == 0:
            score = np.inf
        else:
            q = node.q_value(a)
            score = q + c * np.sqrt(np.log(total_N) / n_sa)

        if score > best_score:
            best_score, best_a = score, a

    return best_a


# ROLLOUT (SIMULATION)
def rollout_from_state(env_sim, max_steps=200):
    total = 0.0
    for _ in range(max_steps):
        a = random.randrange(env_sim.action_space.n)
        _, r, terminated, truncated, _ = env_sim.step(a)
        total += r
        if terminated or truncated:
            break
    return total


# MCTS
def mcts(env_plan, root_state, n_simulations=2, max_rollout_steps=200):
    action_space_n = env_plan.action_space.n
    root = MCTSNode(root_state)

    for _ in range(n_simulations):
        env_sim = copy.deepcopy(env_plan)
        env_sim.unwrapped.state = root_state.copy()

        node = root
        path = []
        done = False

        while True:
            if node.is_terminal:
                done = True
                break

            if not node.is_fully_expanded(action_space_n):
                untried = [a for a in range(action_space_n) if a not in node.children]
                a = random.choice(untried)

                _, r, terminated, truncated, _ = env_sim.step(a)
                done = terminated or truncated

                child_state = env_sim.unwrapped.state.copy()
                child = MCTSNode(child_state, parent=node, action_taken=a)
                child.is_terminal = done

                node.children[a] = child
                node.N.setdefault(a, 0)
                node.W.setdefault(a, 0.0)

                path.append((node, a, r))
                node = child
                break
            else:
                a = select_uct_action(node, action_space_n)
                _, r, terminated, truncated, _ = env_sim.step(a)
                done = terminated or truncated

                path.append((node, a, r))
                node = node.children[a]

                if done:
                    node.is_terminal = True
                    break

        rollout_return = 0.0 if done else rollout_from_state(env_sim, max_rollout_steps)

        G = rollout_return
        for n, a, r in reversed(path):
            G = r + G
            n.N[a] += 1
            n.W[a] += G

    total_visits = sum(root.N.values())
    policy = {a: root.N.get(a, 0) / total_visits for a in root.N} if total_visits > 0 else {}

    return root, policy


def print_tree(node, depth=0, max_depth=3):
    if depth > max_depth:
        return

    indent = "  " * depth
    if node.parent is None:
        print(f"{indent}Root")
    else:
        print(f"{indent}Action taken: {node.action_taken}")

    for a, child in node.children.items():
        q = node.q_value(a)
        n = node.N.get(a, 0)
        print(f"{indent}  ├─ a={a} | Q={q:.2f}, N={n}")
        print_tree(child, depth + 1, max_depth)

def visualize_tree(root, max_depth=3):
    fig, ax = plt.subplots(figsize=(12, 8))
    positions, labels = {}, {}

    def traverse(node, x, y, depth):
        if depth > max_depth:
            return
        node_id = id(node)
        positions[node_id] = (x, y)

        label = []
        for a in node.children:
            label.append(f"a={a}: Q={node.q_value(a):.1f}, N={node.N[a]}")
        labels[node_id] = "\n".join(label) if label else "Leaf"

        dx = 1.0 / (len(node.children) + 1) if node.children else 0
        for i, child in enumerate(node.children.values(), 1):
            cx, cy = x - 0.5 + i * dx, y - 1
            ax.plot([x, cx], [y, cy], 'k-')
            traverse(child, cx, cy, depth + 1)

    traverse(root, 0, 0, 0)

    for nid, (x, y) in positions.items():
        ax.scatter(x, y, s=2200, color="lightblue")
        ax.text(x, y, labels[nid], ha="center", va="center")

    ax.axis("off")
    plt.show()


if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    obs, info = env.reset()

    done = False
    episode_return = 0.0
    last_root = None

    while not done:
        env_plan = gym.make("CartPole-v1")
        env_plan.reset()
        env_plan.unwrapped.state = env.unwrapped.state.copy()

        last_root, policy = mcts(env_plan, env.unwrapped.state.copy(), n_simulations=5)
        action = max(policy, key=policy.get)

        obs, reward, terminated, truncated, _ = env.step(action)
        episode_return += reward
        done = terminated or truncated
        print_tree(last_root)

    print("\nEpisode return:", episode_return)
    print("\nMCTS Tree (Root):")
    # print_tree(last_root, max_depth=3)
    # visualize_tree(last_root, max_depth=3)

    env.close()
