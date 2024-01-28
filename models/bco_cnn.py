###### Define student agent
import torch 
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)

from os.path import join, exists
from os import mkdir, remove, rename

import torch.nn.functional as F

# from os import mkdir, unlink, listdir, getpid, remove


class BCO_cnn(nn.Module):

    def __init__(self, obs_space, action_space, h_size=16, scaler=None, device='cpu') -> None:
        
        super(BCO_cnn,self).__init__()

        self.name = 'bco-cnn'
        self.scaler = scaler
        self.device = device

        self.n_inputs = obs_space
        self.n_outputs = action_space

        # [197,1]
        self.conv1 = nn.Conv1d(in_channels=1,out_channels=36,kernel_size=5,stride=2,padding=1)
        self.conv2 = nn.Conv1d(in_channels=36,out_channels=36,kernel_size=3,stride=2,padding=2)
        self.fc1 = nn.Linear(in_features=1800,out_features=72)
        self.fc2 = nn.Linear(in_features=72,out_features=self.n_outputs)
        self.LRelu = nn.LeakyReLU()


    def forward(self, x):
        # print(x.shape)
        x = self.conv1(x)
        # print(x.shape)
        x = self.conv2(x)
        # print(x.shape)
        x = x.view(x.size(0),-1)
        # print(x.shape)
        x = self.LRelu(self.fc1(x))
        # print(x.shape)
        x = self.fc2(x)
        # print(x.shape)
        return x

    def act(self, state):
        n_obs = self.scaler.transform(state.reshape(1, -1))
        obs = torch.from_numpy(n_obs).float() #.unsqueeze(0) # [1,196]
        a = self.forward(obs).squeeze().detach().cpu().numpy()
        obs = obs.unsqueeze(1)
        return a

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
            if exists(dest+save_name):rename(dest+save_name, dest+save_name+'.bk')

        torch.save(self.state_dict(), dest+save_name)





#Test
# prova = torch.rand(197,1)
# network = BCO_cnn(197,36)
# network.forward(prova.T.unsqueeze(0))












