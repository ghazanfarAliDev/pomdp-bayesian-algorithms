import numpy as np
import random
import copy
from mcts_node import MCTSNode


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
def mcts(env_plan, root_state, n_simulations=50, max_rollout_steps=200):
    action_space_n = env_plan.action_space.n
    root = MCTSNode(root_state)

    for _ in range(n_simulations):
        env_sim = copy.deepcopy(env_plan)
        env_sim.reset()
        env_sim.state = root_state.copy() if hasattr(env_sim, 'state') else root_state.copy()

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

                obs, r, terminated, truncated, _ = env_sim.step(a)
                done = terminated or truncated

                child_state = np.array(env_sim.state if hasattr(env_sim, 'state') else obs)
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
                obs, r, terminated, truncated, _ = env_sim.step(a)
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


# Optional: print tree for debugging
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
