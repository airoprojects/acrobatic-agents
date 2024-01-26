'''
Note: this code is part of the bullet3 library: (https://github.com/bulletphysics/bullet3/tree/master)
This script HAS been modified:
  1. Imported custom PPO Agent
'''


import json
import numpy as np
import pybullet_data

# import custom ppo agent
from deep_mimic.ppo_agent import PPOAgent

AGENT_TYPE_KEY = "AgentType"


def build_agent(world, id, file):
  agent = None
  with open(pybullet_data.getDataPath() + "/" + file) as data_file:
    json_data = json.load(data_file)

    assert AGENT_TYPE_KEY in json_data
    agent_type = json_data[AGENT_TYPE_KEY]

    if (agent_type == PPOAgent.NAME):
      agent = PPOAgent(world, id, json_data)
      
    else:
      assert False, 'Unsupported agent type: ' + agent_type

  return agent
