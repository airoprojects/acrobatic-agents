import sys
import time
from git import Repo

import torch
import numpy as np 

from model import BCAgent
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

  update_timestep = 1. / 240.
  animating = True
  step = False
  total_reward = 0
  steps = 0
  args = sys.argv[1:]

  actions = []
  observations = []

  world = dm.build_world(args, True, enable_stable_pd=True)

  policy = BCAgent(196,36,device).eval()

  src = root_dir+'/checkpoints/'+policy.name.lower()+'.pt'
  policy.load_parameters(src)
  
  
  min_val = -61.59686279296875

  max_val = 68.45513916015625

  while (world.env._pybullet_client.isConnected()):

    timeStep = update_timestep
    time.sleep(timeStep)
    keys = world.env.getKeyboardEvents()

    if world.env.isKeyTriggered(keys, ' '):
      animating = not animating
   
    if world.env.isKeyTriggered(keys, 'i'):
      step = True
   
    if (animating or step):
      s, a = dm.update_world(world, timeStep, override=(policy, min_val, max_val)) 
      step = False