# POMDP Bayesian MCTS

## About the Project

This project implements a **Monte Carlo Tree Search (MCTS)** algorithm for the **CartPole environment** using **JAX** for fast, JIT-compiled computation. The MCTS algorithm is fully batched and compatible with JAX’s functional programming style, allowing high-performance simulations on CPU or GPU.

Key features:
- Batched MCTS simulation for multiple parallel environments.
- JIT-compiled rollout and selection for fast execution.
- Full tree expansion, selection, rollout, and backpropagation.
- Visualizes the MCTS tree for inspection of visit counts and policies.
- Configurable parameters for batch size, number of simulations, rollout depth, and UCT exploration.

---

## Project Structure

pomdp-bayesian-algorithms/
    - main.py # Entry point for running the MCTS experiment 
    -config.py # Hyperparameters like RNG seed, batch size, max simulations
envs/
    - cartpole_env.py # CartPole environment wrapper using Gymnax
mcts/
    - tree.py # Tree initialization
    - rollout.py # Rollout function
    - mcts_step.py # Single MCTS step (selection, expansion, rollout, backprop)
    - visualize_tree.py # Optional tree visualization using NetworkX and matplotlib
    - requirements.txt # Python dependencies

## Installation Instructions

Follow these steps to set up the project on **Windows**:

1. Install Python 3.11

Check if Python 3.11 is installed:
    - py -3.11 --version

2. Create a virtual environment
    - cd project_path
    - py -3.11 -m venv venv

3. Activate the virtual environment
    - venv\Scripts\activate

4. Install Dependencies:
    - pip install -r requirements.txt