###### Define student agent
import torch 
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)

from os.path import join, exists
# from os import mkdir, unlink, listdir, getpid, remove


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class BCAgent(nn.Module):

  def __init__(self, obs_space, action_space) -> None:
    super(BCAgent,self).__init__()
    self.name = 'Behavioral-Cloning-Agent'

    self.n_inputs = obs_space
    self.n_outputs = action_space

    # Policy Network
    self.fc1 = nn.Linear(self.n_inputs,16)
    self.bn1 = nn.BatchNorm1d(16)
    self.relu = nn.ReLU()
    self.fc2 = nn.Linear(16, self.n_outputs)

  def forward(self, x):
    out = self.fc1(x)
    out = self.bn1(out)
    out = self.relu(out)
    out = self.fc2(out)
    return out
  
  def load_parameters(self, dir):
    # if exists(dir+self.name.lower()+'.pt'): 
    if exists(dir+self.name.lower()+'new'+'.pt'):
        print("Loading model "+self.name+" state parameters")
        self.load_state_dict(torch.load(dir+self.name.lower()+'.pt', map_location=self.device))
        return self
    else:
        print("Error no model "+self.name.lower()+" found!")
        exit(1)

  


