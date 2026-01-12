import time
import gymnasium as gym
from mcts_node import MCTSNode
from mcts_utils import mcts, print_tree

env = gym.make("CartPole-v1", render_mode="rgb_array")
obs, info = env.reset(seed=42)

done = False
episode_return = 0.0
last_root = None

start_episode = time.time()

while not done:
    env_plan = gym.make("CartPole-v1")
    obs_plan, _ = env_plan.reset(seed=42)
    root_state = obs if obs is not None else obs_plan  # use observation as root state

    last_root, policy = mcts(env_plan, root_state, n_simulations=2000)

    if not policy:
        action = env.action_space.sample()  # fallback if policy is empty
    else:
        action = max(policy, key=policy.get)

    obs, reward, terminated, truncated, _ = env.step(action)
    episode_return += reward
    done = terminated or truncated
end_episode = time.time()

print(f"\nEpisode return: {episode_return}")
print(f"Total episode took {end_episode - start_episode:.4f} seconds")

# print("\nMCTS Tree (Root):")
# print_tree(last_root, max_depth=2)

env.close()
