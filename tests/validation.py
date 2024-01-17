import gymnasium as gym
env = gym.make("Humanoid-v4", render_mode="human", xml_file="humanoid_test.xml" )
observation, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

env.close()