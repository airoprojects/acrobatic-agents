import os
import sys
import time
import json
import inspect
import argparse
from git import Repo

import pybullet_data
from pybullet_utils.logger import Logger
from pybullet_utils.arg_parser import ArgParser

import random
import numpy as np

# import ..deep_mimic.rl_util as dm
# from deep_mimic.rl_world import RLWorld  
# from deep_mimic.ppo_agent import PPOAgent 
# from deep_mimic.pybullet_deep_mimic_env import PyBulletDeepMimicEnv

# setup project root dir
repo = Repo(".", search_parent_directories=True)
root_dir = repo.git.rev_parse("--show-toplevel")
print("root: {}".format(root_dir))

# Get the path to this file
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir+'/deep_mimic/')
print(sys.path)
from rl_util import build_world 
from rl_util import update_world


# Add the root directory to sys.path if it's not already there
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

print(sys.path)

repo = Repo(".", search_parent_directories=True)
root_dir = repo.git.rev_parse("--show-toplevel")
print("root: {}".format(root_dir))

if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument('-i', '--iterations', type=int, help="Insert num of rollouts")
  parser.add_argument('-t', '--task', type=str, help="Insert type of task : avaliable [backflip, spinkick])")
  args = parser.parse_args()

  # Set up the environment
  num_interactions = args.iterations if args.iterations else 6000
  task_type = args.task if args.task else 'backflip'

  update_timestep = 1. / 240.
  animating = True
  step = False
  # args = sys.argv[1:]

  # env
  world = build_world(True, enable_stable_pd=True,task = task_type)
  obs_dim = world.env.get_state_size()
  action_dim = world.env.get_action_size()

  # data collection np arrays
  step_counter = 0
  discarded = 0
  observations = np.empty((num_interactions,) + (obs_dim,)) # 197
  actions = np.empty((num_interactions,) + (action_dim,)) # 36

  while (world.env._pybullet_client.isConnected()):

    timeStep = update_timestep
    time.sleep(timeStep)
    keys = world.env.getKeyboardEvents()

    if world.env.isKeyTriggered(keys, ' '):
      animating = not animating
   
    if world.env.isKeyTriggered(keys, 'i'):
      step = True
   
    if (animating or step):
      s, a = update_world(world, timeStep, override=False) 
      step = False

      if step_counter >= num_interactions: break
      
      if (not s is None) and (not a is None):
        if (a > 1e2).any() or (a > 1e2).any():
          discarded +=1
          continue

        else:
            actions[step_counter] = a
            observations[step_counter] = s
            step_counter += 1
      else:
        discarded += 1

  # observations = np.asarray(observations, dtype=object)
  # actions = np.asarray(actions, dtype=object)
  # currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        
  print('discarded', discarded)

  # saving
  save_dir = root_dir+'/data/'+str(task_type)
  if not os.path.exists(save_dir):
    os.makedirs(save_dir)

  np.save(save_dir+'/expert-observations-'+str(num_interactions), observations) 
  np.save(save_dir+'/expert-actions-'+str(num_interactions), actions) 

  # np.save(root_dir+'/data/expert-observations-'+str(type_task)+'-'+str(num_interactions), observations) 
  # np.save(root_dir+'/data/expert-actions-'+str(type_task)+'-'+str(num_interactions), actions) 

