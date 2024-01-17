import gymnasium as gym
# from stable_baselines3 import PPO

import numpy as np

import sys
from git import Repo

# Initialize the Git repository object
repo = Repo(".", search_parent_directories=True)

# Get the root directory of the Git project
root_dir = repo.git.rev_parse("--show-toplevel")

# Add custom modules to path
datasets_path = root_dir  + '/datasets/'
sys.path.insert(0, datasets_path)

# Create the environment
env = gym.make("Humanoid-v4", render_mode="human")

# Extract actions from expert policy
expert_actions = np.load(datasets_path+'1-expert-data/expert_actions.npy')
# expert_actions = np.load('expert_actions.npy')

# Test the trained agent
num_interactions = 10000 #int(4e4)
expert_observation, info = env.reset()
for i in range(num_interactions):
    expert_action = expert_actions[i]
    expert_observation, reward, terminated, truncated, info = env.step(expert_action)

    if terminated or truncated:
        expert_observation, info = env.reset()

env.close()

