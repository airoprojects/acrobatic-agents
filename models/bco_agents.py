###### Define student agent
import torch 
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)

from os.path import join, exists
from os import mkdir, remove, rename
# from os import mkdir, unlink, listdir, getpid, remove


class BCOAgentFC(nn.Module):

  def __init__(self, obs_space, 
               action_space,
               h_size=16,
               device='cpu'
              ) -> None:
    
    super(BCOAgentFC, self).__init__()

    self.name = 'bco-fc'
    self.device = device

    self.n_inputs = obs_space
    self.n_outputs = action_space

    # Policy Network
    self.fc1 = nn.Linear(self.n_inputs, h_size) #16
    self.bn1 = nn.BatchNorm1d(h_size) #16
    self.relu = nn.ReLU()
    self.fc2 = nn.Linear(h_size, self.n_outputs) #16

  def forward(self, x):
    out = self.fc1(x)
    out = self.bn1(out)
    out = self.relu(out)
    out = self.fc2(out)
    return out
  
  def load_parameters(self, src, version):
    src = src+self.name.lower()+'-'+str(version)+'.pt'
    if exists(src):
        print("Loading model "+self.name.lower()+'-'+str(version)+" state parameters")
        print("From :{}".format(src))
        self.load_state_dict(torch.load(src, map_location=self.device))
        return self
    else:
        print("Error no model "+self.name.lower()+'-'+str(version)+'.pt'+" found!")
        exit(1)

  
  def save_parameters(self, dest, version):
    save_name = self.name.lower()+'-'+str(version)+'.pt'

    if not exists(dest): 
      mkdir(dest)
    else: 
        if exists(dest+save_name):
          rename(dest+save_name, dest+save_name+'.bk')

    torch.save(self.state_dict(), dest+save_name)


