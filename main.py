import sys
import time
import pickle
import joblib
import argparse
from git import Repo

import torch
import numpy as np 

from models import BCOAgentFC
from models import BCO_cnn
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
  parser.add_argument('-v', '--version', type=str, help="Insert model version ")
  parser.add_argument('-t', '--task', type=str, help="Insert type of task(backflip, spinkick, ... )")


  my_args = parser.parse_args()
  type_task = my_args.task if my_args.task else 'backflip'


  # env set up
  update_timestep = 1. / 240.
  animating = True
  step = False
  args = sys.argv[1:]

  # actions = []
  # observations = []

  # env
  world = dm.build_world(args, True, enable_stable_pd=True, task=type_task)

  # scaler
  scaler_version = 20000 #'mixed-6000'
  scaler_path = root_dir+'/data/scaler-'+str(scaler_version)+'.joblib'
  scaler = joblib.load(scaler_path)    
  min_val = -61.59686279296875
  max_val = 68.45513916015625

  # model 
  obs_dim = world.env.get_state_size()
  action_dim = world.env.get_action_size()

  version = my_args.version if my_args.version else 100
  # policy = BCOAgentFC(obs_dim, action_dim, h_size=obs_dim*2, scaler=scaler, device=device).eval()
  policy = BCO_cnn(obs_dim, action_dim, scaler=scaler).eval()

  src = root_dir+'/checkpoints/'
  policy.load_parameters(src, version=version)


  # override = {
  #   'policy': policy,
  #   'min': min_val,
  #   'max': max_val,
  #   'scaler': scaler
  # }

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
