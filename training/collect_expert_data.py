import os
import sys
import time
import json
import inspect
from git import Repo
import pybullet_data
from pybullet_utils.logger import Logger
from pybullet_utils.arg_parser import ArgParser

# setup project root dir
repo = Repo(".", search_parent_directories=True)
root_dir = repo.git.rev_parse("--show-toplevel")
print("root: {}".format(root_dir))

# Get the root of the project
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Add the root directory to sys.path if it's not already there
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

print(sys.path)

repo = Repo(".", search_parent_directories=True)
root_dir = repo.git.rev_parse("--show-toplevel")
print("root: {}".format(root_dir))


# from pybullet_envs.deep_mimic.learning.rl_world import RLWorld
# from pybullet_envs.deep_mimic.learning.ppo_agent import PPOAgent
# from pybullet_envs.deep_mimic.env.pybullet_deep_mimic_env import PyBulletDeepMimicEnv

from deep_mimic.rl_world import RLWorld  
from deep_mimic.ppo_agent import PPOAgent 
from deep_mimic.pybullet_deep_mimic_env import PyBulletDeepMimicEnv

# Now you can use RLAgent and RLWorld in your collect-expert-data.py


import random
import numpy as np

update_timestep = 1. / 240.
animating = True
step = False
total_reward = 0
steps = 0

def update_world(world, time_elapsed):
  timeStep = update_timestep
  s, a = world.update(timeStep)
  reward = world.env.calc_reward(agent_id=0)
  global total_reward
  total_reward += reward
  global steps
  steps+=1
  
  #print("reward=",reward)
  #print("steps=",steps)
  end_episode = world.env.is_episode_end()
  if (end_episode or steps>= 1000):
    print("total_reward=",total_reward)
    total_reward=0
    steps = 0
    world.end_episode()
    world.reset()

  return s, a

def build_arg_parser(args):
  arg_parser = ArgParser()
  arg_parser.load_args(args)

  arg_file = arg_parser.parse_string('arg_file', '')
  if arg_file == '':
    arg_file = "run_humanoid3d_backflip_args.txt"
  if (arg_file != ''):
    path = pybullet_data.getDataPath() + "/args/" + arg_file
    succ = arg_parser.load_file(path)
    Logger.print2(arg_file)
    assert succ, Logger.print2('Failed to load args from: ' + arg_file)
  return arg_parser

args = sys.argv[1:]

def build_world(args, enable_draw):

  arg_parser = build_arg_parser(args)
  print("enable_draw=", enable_draw)
  env = PyBulletDeepMimicEnv(arg_parser, enable_draw)
  world = RLWorld(env, arg_parser)
  #world.env.set_playback_speed(playback_speed)

  motion_file = arg_parser.parse_string("motion_file")
  print("motion_file=", motion_file)
  bodies = arg_parser.parse_ints("fall_contact_bodies")
  print("bodies=", bodies)
  int_output_path = arg_parser.parse_string("int_output_path")
  print("int_output_path=", int_output_path)
  agent_files = pybullet_data.getDataPath() + "/" + arg_parser.parse_string("agent_files")

  AGENT_TYPE_KEY = "AgentType"

  print("agent_file=", agent_files)
  with open(agent_files) as data_file:
    json_data = json.load(data_file)
    print("json_data=", json_data)
    assert AGENT_TYPE_KEY in json_data
    agent_type = json_data[AGENT_TYPE_KEY]
    print("agent_type=", agent_type)
    agent = PPOAgent(world, id, json_data)

    agent.set_enable_training(False)
    world.reset()
    
  return world

if __name__ == '__main__':

  actions = []
  observations = []
  step_counter = 0
  discarded = 0

  num_interactions = 6000

  observations = np.empty((num_interactions,) + (196,))
  actions = np.empty((num_interactions,) + (36,))

  world = build_world(args, True)
  while (world.env._pybullet_client.isConnected()):

    timeStep = update_timestep
    time.sleep(timeStep)
    keys = world.env.getKeyboardEvents()

    if world.env.isKeyTriggered(keys, ' '):
      animating = not animating
   
    if world.env.isKeyTriggered(keys, 'i'):
      step = True
   
    if (animating or step):
      s, a = update_world(world, timeStep) #this is modded
      step = False

      if step_counter >= num_interactions: break
      
      if not (s is None and a is None):
        if (a > 1e2).any() or (a > 1e2).any():
          discarded +=1
          continue
        else:
            actions[step_counter] = a
            observations[step_counter] = s[:196]

      step_counter += 1



  # observations = np.asarray(observations, dtype=object)
  # actions = np.asarray(actions, dtype=object)
  
  currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

  print('discarded', discarded)
  np.save(root_dir+'/data/expert-observations', observations) 
  np.save(root_dir+'/data/expert-actions', actions) 

