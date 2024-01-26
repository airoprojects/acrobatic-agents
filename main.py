import sys
import time

import numpy as np 

import deep_mimic.rl_util as dm
from model import BCAgent

if __name__ == '__main__':

  update_timestep = 1. / 240.
  animating = True
  step = False
  total_reward = 0
  steps = 0
  args = sys.argv[1:]

  actions = []
  observations = []

  world = dm.build_world(args, True)

  policy = BCAgent(196,36).eval()

  while (world.env._pybullet_client.isConnected()):

    timeStep = update_timestep
    time.sleep(timeStep)
    keys = world.env.getKeyboardEvents()

    if world.env.isKeyTriggered(keys, ' '):
      animating = not animating
   
    if world.env.isKeyTriggered(keys, 'i'):
      step = True
   
    if (animating or step):
      s, a = dm.update_world(world, timeStep, update_timestep, override=policy) 
      step = False