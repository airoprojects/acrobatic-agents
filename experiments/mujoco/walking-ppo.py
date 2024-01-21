import gymnasium as gym

import numpy as np

import sys
from tqdm import tqdm
from git import Repo

# Initialize the Git repository object
repo = Repo(".", search_parent_directories=True)

# Get the root directory of the Git project
root_dir = repo.git.rev_parse("--show-toplevel")

# Add custom modules to path
datasets_path = root_dir  + '/data/datasets/'
sys.path.insert(0, datasets_path)

# Create the environment
env = gym.make("Humanoid-v4", render_mode="human")

# Extract actions from expert policy
expert_actions = np.load(datasets_path+'expert-data-3/expert-actions.npy')
expert_observations = np.load(datasets_path+'expert-data-2/expert-observations.npy')
# expert_actions = np.load('expert_actions.npy')

# Test the trained agent
num_interactions = 10000 #int(4e4)
observation, info = env.reset()
current_observations = np.empty((num_interactions,) + env.observation_space.shape)

for i in tqdm(range(num_interactions)):
    expert_action = expert_actions[i]
    observation, reward, terminated, truncated, info = env.step(expert_action)
    current_observations[i] = observation

    # print(info)

    if terminated or truncated:
        expert_observation, info = env.reset()

env.close()


