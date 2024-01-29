import sys
import time
import pickle
import joblib
import argparse
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
  parser.add_argument('-m', '--model', type=int, help="Insert one model of you choice: [1: fc, 2:cnn]")
  parser.add_argument('-v', '--version', type=str, help="Insert model version")
  parser.add_argument('-s', '--scaler', type=str, help="Insert scaler version")
  parser.add_argument('-t', '--task', type=str, help="Insert type of task : avaliable [backflip, spinkick])")
  args = parser.parse_args()

  model_type = args.model if args.model else 1
  model_version = args.version if args.version else 1
  scaler_version = args.scaler if args.scaler else 20000
  task_type = args.task if args.task else 'backflip'


  # env set up
  update_timestep = 1. / 240.
  animating = True
  step = False
  # args = sys.argv[1:]

  # env
  world = dm.build_world(True, enable_stable_pd=True, task=task_type)

  # scaler
  scaler_path = root_dir+'/data/scaler-'+str(scaler_version)+'.joblib'
  scaler = joblib.load(scaler_path)    

  # model 
  obs_dim = world.env.get_state_size()
  action_dim = world.env.get_action_size()
  
  if model_type == 1:
    policy = BCOAgentFC(obs_dim, action_dim, h_size=obs_dim*2, scaler=scaler, device=device).eval()
  
  elif model_type == 2:
    policy = BCOCNN(obs_dim, action_dim, scaler=scaler).eval()

  else:
    raise("Model does not exist ")

  # load policy parameters
  src = root_dir+'/checkpoints/'+str(task_type)+'/'
  policy.load_parameters(src, version=model_version)

  # simulation
  while (world.env._pybullet_client.isConnected()):

    timeStep = update_timestep
    time.sleep(timeStep)
    keys = world.env.getKeyboardEvents()

    if world.env.isKeyTriggered(keys, ' '):
      animating = not animating
   
    if world.env.isKeyTriggered(keys, 'i'):
      step = True
   
    if (animating or step):
      s, a = dm.update_world(world, timeStep, override=policy) 
      step = False
