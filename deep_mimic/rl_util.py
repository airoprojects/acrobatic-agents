'''
Note: this code is part of the bullet3 library: (https://github.com/bulletphysics/bullet3/tree/master)
This script HAS been modified:
  1. Added 3 new function to be used to run a simulation:
      - update update_world(world, time_elapsed, update_timestep, override=False)
      - build_arg_parser(args)
      - build_world(args, enable_draw)
  2. Note: the global variable SHOULD be removed form here (but is a minor issue to resolve later)
'''

import numpy as np

import os
import sys
import time
import json
import inspect

import pybullet_data
from pybullet_utils.logger import Logger
from pybullet_utils.arg_parser import ArgParser

# Get the root of the project
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Add the root directory to sys.path if it's not already there
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from deep_mimic.rl_world import RLWorld  
from deep_mimic.ppo_agent import PPOAgent 
from deep_mimic.pybullet_deep_mimic_env import PyBulletDeepMimicEnv

# global variable
total_reward = 0
steps = 0


def update_world(world, time_elapsed, override=False):
  timeStep = time_elapsed
  s, a = world.update(timeStep, override=override)

  reward = world.env.calc_reward(agent_id=0)
  global total_reward
  total_reward += reward
  #print("reward=",reward)

  global steps
  steps+=1
  #print("steps=",steps)
  
  end_episode = world.env.is_episode_end()
  if (end_episode or steps>= 5000):
    print("total_reward=",total_reward)
    total_reward=0
    steps = 0
    world.end_episode()
    world.reset()

  return s, a

def build_arg_parser(task=None):
  arg_parser = ArgParser()
  arg_file = ''
  # arg_parser.load_args(args)
  # arg_file = arg_parser.parse_string('arg_file', '')

  # add here other mocap data if you want change:
  if task == 'spinkick':
    # arg_file = arg_parser.parse_string('arg_file', 'run_humanoid3d_spinkick_args.txt')
    arg_file = "run_humanoid3d_spinkick_args.txt"
  
  if task == 'backflip':
    arg_file = "run_humanoid3d_backflip_args.txt"

  if task == 'jump':
    arg_file = "run_humanoid3d_jump_args.txt"

  if (arg_file != ''):
    path = pybullet_data.getDataPath() + "/args/" + arg_file
    succ = arg_parser.load_file(path)
    Logger.print2(arg_file)
    assert succ, Logger.print2('Failed to load args from: ' + arg_file)

  else:
    print('Task not available !!!')
    return None

  return arg_parser


def build_world(enable_draw, enable_stable_pd=True, task=None):
  arg_parser = build_arg_parser(task)
  print("enable_draw=", enable_draw)

  env = PyBulletDeepMimicEnv(arg_parser, enable_draw, enable_stable_pd=enable_stable_pd)
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

def compute_return(rewards, gamma, td_lambda, val_t):
  # computes td-lambda return of path
  path_len = len(rewards)
  assert len(val_t) == path_len + 1

  return_t = np.zeros(path_len)
  last_val = rewards[-1] + gamma * val_t[-1]
  return_t[-1] = last_val

  for i in reversed(range(0, path_len - 1)):
    curr_r = rewards[i]
    next_ret = return_t[i + 1]
    curr_val = curr_r + gamma * ((1.0 - td_lambda) * val_t[i + 1] + td_lambda * next_ret)
    return_t[i] = curr_val

  return return_t
