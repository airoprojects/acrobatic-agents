import gymnasium as gym
import numpy as np

def simple_walking_policy(observation):
    # This is a very basic and naive policy, just for demonstration.
    # It oscillates the joints within a mid-range to mimic a walking pattern.
    # In practice, you'd need a much more sophisticated approach, likely using RL.
    
    # Assuming the action space is continuous and symmetric around zero
    joint_angles = observation[:env.action_space.shape[0]]
    action = np.sin(joint_angles)
    return action

wait = 40
jump = 80

def simple_jump_policy(observation, step):
    action = np.zeros(env.action_space.shape[0])

    # Simplified logic for bending and extending legs
    if step < wait:
        # action = np.zeros(17, dtype=np.float32)

        # flex torso forward
        action[0] = -0.2

        # # bens elbows
        action[13] = 0.2
        action[16] = 0.2
            

    elif wait <= step < jump:  

        # push legs       
        action[6] = 0.5
        action[10] = 0.5 

        # flex torso backward
        action[0] = 0.4

        # # push shoulders
        # # action[12] = 0.4
        # # action[15] = 0.4

        # extend elbows
        action[13] = -0.4
        action[16] = -0.4

      
    elif step >= jump:
         action = np.zeros(17, dtype=np.float32)

    # print(action)
    return action

env = gym.make("Humanoid-v4", render_mode="human")
observation, info = env.reset()

step = 0

for _ in range(1000):
    # action = simple_walking_policy(observation)  
    action = simple_jump_policy(observation, step) 
    observation, reward, terminated, truncated, info = env.step(action)

    # if terminated: #or truncated:
    #     observation, info = env.reset()
    #     step = 0

    if step >= 100:
        env.reset()  
        step = 0 
    
    step += 1

env.close()