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

wait = 35 #step of iteration that must wait
jump = 70

def simple_jump_policy(observation, step):
    action = np.zeros(env.action_space.shape[0]) 


    # Simplified logic for bending and extending legs
    if step < wait:
        #do nothing
        action = np.zeros(17, dtype=np.float32)
        # 17 -> number of joints

    elif wait <= step < jump:  # Bend legs
        # action[4] = 0.4  
        # action[8] = 0.4 

        action[6] = 0.4  # torque max to joint left
        action[10] = 0.4 # torque max to joint right

    # elif 55 <= step < 80:  # Extend legs
    #     action[4] = 0.4  
    #     action[8] = 0.4 

    #     action[6] = 0.4  
    #     action[10] = 0.4 

      
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