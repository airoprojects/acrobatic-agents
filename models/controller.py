import torch
import torch.nn as nn
torch.manual_seed(42)

from os.path import join, exists
from os import mkdir, unlink, listdir, getpid, remove, rename


class Controller(nn.Module):
    """ Controller """

    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.name = 'CONTROLLER'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # continuous
        self.fc= nn.Linear(in_dim, out_dim)
        self.th = nn.Tanh()


    def forward(self, c_in):
        out = self.fc(c_in)
        # out = self.th(out)
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