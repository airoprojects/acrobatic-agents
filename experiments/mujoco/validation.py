import gymnasium as gym
import argparse

parser = argparse.ArgumentParser()
env_id_list = ["Humanoid-v4", "HumanoidStandup-v4"]
parser.add_argument('--env', type=str, help="Insert gymnasium env_id: {}".format(env_id_list))
args = parser.parse_args()

env_id = args.env if args.env else 'Humanoid-v4'

env = gym.make(env_id, render_mode="human")
observation, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

env.close()