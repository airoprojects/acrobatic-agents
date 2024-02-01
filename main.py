import os
import sys
import time
import json
import pickle
import joblib
import argparse
# import argcomplete
from git import Repo

import torch
import numpy as np 

from models import BCOAgentFC
from models import BCOCNN
import deep_mimic.rl_util as dm

# set up train device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'
print("device: {}".format(device))

# setup project root dir
repo = Repo(".", search_parent_directories=True)
root_dir = repo.git.rev_parse("--show-toplevel")
print("root: {}".format(root_dir))

if __name__ == '__main__':

  # arg pareser
  parser = argparse.ArgumentParser()
  parser.add_argument('--info', action='store_true', help=f"Version lookup table.")
  parser.add_argument('-m', '--model', type=str, help=f"Insert model type.")
  parser.add_argument('-v', '--version', type=str, help="Insert model version.")
  parser.add_argument('-s', '--scaler', type=str, help="Insert scaler version. Default 20000 (best).")
  parser.add_argument('-t', '--task', type=str, help=f"Insert type of task")
  args = parser.parse_args()

  # info
  if args.info:

    print("\n\nUsage: python main.py [-h] [--info] [-m MODEL] [-v VERSION] [-s SCALER] [-t TASK]\n")

    with open(root_dir+'/versions/model_versions.json', 'r') as file:
      model_versions = json.load(file)
  
    table_str = "Available Models and Versions:\n\n"
    table_str += "Model            | Versions\n"
    table_str += "-----------------|-----------------\n"

    for model, versions in model_versions.items():
        versions_str = ", ".join(versions)
        table_str += f"{model:<16} | {versions_str}\n"

    print(table_str)
    available_tasks = ['backflip', 'spinkick', 'jump', 'mixed']

    print("Scalers:")
    items = set()
    for data_task in available_tasks:
      contents = os.listdir(root_dir+'/data/'+data_task)
      for item in contents:
          if item not in items and 'scaler' in item: print(item)
          items.add(item)

    print("\nAvailable tasks: ")
    for task in available_tasks:
      print(task)
    print()
    exit()

  model_type = args.model if args.model else 'bco-cnn-backflip'
  model_version = args.version if args.version else '20k'
  scaler_version = args.scaler if args.scaler else 20000
  task_type = args.task if args.task else 'backflip'
  
  # scaler
  scaler_type = model_type.split('-')[2]
  scaler_path = root_dir+'/data/'+str(scaler_type)+'/scaler-'+str(scaler_version)+'.joblib'
  scaler = joblib.load(scaler_path)    

  if task_type == 'mixed':
    if np.random.uniform(low=0.0, high=1.0) > 0.5: 
      task_type = 'spinkick'
    else:
      task_type = 'backflip'

  # env set up
  update_timestep = 1. / 240.
  animating = True
  step = False
  # args = sys.argv[1:]

  # env
  world = dm.build_world(enable_draw=True, task=task_type)

  # env dim
  obs_dim = world.env.get_state_size()
  action_dim = world.env.get_action_size()
  
  print("\n"+"#"*10)
  print("Environment set for task: {}".format(task_type))
  print("Model type: {}".format(model_type))
  print("Model version: {}".format(model_version))
  print("Scaler version: {}".format(scaler_version))

  # model 
  if 'bco-fc' in model_type:
    policy = BCOAgentFC(obs_dim, action_dim, h_size=obs_dim*2, scaler=scaler, device=device).eval()
  
  elif 'bco-cnn' in model_type:
    policy = BCOCNN(obs_dim, action_dim, scaler=scaler).eval()

  else:
    raise("Model does not exist ")

  # load policy parameters
  task_dir = model_type.split('-')[2]
  src = root_dir+'/checkpoints/'+str(task_dir)+'/'
  policy.load_parameters(src, version=model_version)
  print("#"*10+"\n")

  print("Starting simulation: \npress q for quit \npress space for pause\n")

  # simulation
  while (world.env._pybullet_client.isConnected()):

    timeStep = update_timestep
    time.sleep(timeStep)
    keys = world.env.getKeyboardEvents()

    if world.env.isKeyTriggered(keys, ' '):
      animating = not animating
   
    if world.env.isKeyTriggered(keys, 'i'):
      step = True    
    
    if world.env.isKeyTriggered(keys, 'q'):
      break
   
    if (animating or step):
      s, a = dm.update_world(world, timeStep, override=policy) 
      step = False
