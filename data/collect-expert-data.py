import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.type_aliases import Schedule

import numpy as np

import os
import sys
import argparse
from os import path
from git import Repo
from tqdm import tqdm

# setup project root dir
repo = Repo(".", search_parent_directories=True)
root_dir = repo.git.rev_parse("--show-toplevel")

# Add datasets to path
datasets_path = root_dir  + '/data/datasets/'
sys.path.insert(0, datasets_path)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    env_id_list = ["Humanoid-v4", "HumanoidStandup-v4"]
    parser.add_argument('--env', type=str, help="Insert gymnasium env_id: {}".format(env_id_list))
    parser.add_argument('-s', '--save', type=str, help="Insert path to save expert data")
    parser.add_argument('-n', '--name', type=str, help="Insert the name you want to call the dataset")
    args = parser.parse_args()

    # Set up the environment
    env_id = args.env if args.env else 'Humanoid-v4'
    env = gym.make(env_id, render_mode="rgb_array")

    # Instantiate the agent
    expert_model = PPO(
        "MlpPolicy", 
        env_id,
        n_steps=int(4e4),
        n_epochs=100, 
        verbose=1
    )

    # Train the agent
    expert_model.learn(
        total_timesteps=3e4
    )

    # Evaluate expert
    mean_reward, std_reward = evaluate_policy(expert_model, Monitor(env), n_eval_episodes=10)
    print(f"Mean reward expert agent= {mean_reward} +/- {std_reward}")

    #TODO: add control to save data only if evaluation is good enough

    # Empty dataset
    num_interactions = int(4e4)

    expert_observations = np.empty((num_interactions,) + env.observation_space.shape)
    expert_actions = np.empty((num_interactions,) + env.action_space.shape)

    print(expert_observations.shape)
    print(expert_actions.shape)

    # Collect experience usign expert policy
    obs, _ = env.reset()
    for i in tqdm(range(num_interactions)):
        action, _ = expert_model.predict(obs, deterministic=True)
        expert_observations[i] = obs    # save observations
        expert_actions[i] = action      # save actions
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        if done:
            obs, _ = env.reset()


    # Save dataset
            
    dataset_name = args.name if args.name else 'expert-data'
    dafault_path = datasets_path
    save_path = args.save if args.save else dafault_path 
    save_path += dataset_name

    print("Saving data in: {}".format(save_path))

    if path.exists(save_path) == False:
      print("Destination folder has been created")
      os.makedirs(save_path)


    np.save(save_path+'/expert-actions', expert_actions) 
    np.save(save_path+'/expert-observations', expert_observations) 

    # np.savez_compressed(
    # save_path,
    # expert_actions=expert_actions,
    # expert_observations=expert_observations,
    # 