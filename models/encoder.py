
import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(42)

from os.path import join, exists
from os import mkdir, unlink, listdir, getpid, remove, rename

LATENT = 72
CHANNELS = 1
OBS_SIZE = 197

###### Define student agent
import torch 
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)

from os.path import join, exists
from os import mkdir, remove, rename

import torch.nn.functional as F

# from os import mkdir, unlink, listdir, getpid, remove


class Encoder(nn.Module):

    def __init__(self, obs_space, action_space, h_size=16, scaler=None, device='cpu') -> None:
        
        super(Encoder,self).__init__()

        self.name = 'bco-cnn'
        self.scaler = scaler
        self.device = device

        self.n_inputs = obs_space
        self.n_outputs = action_space

        # [197,1]
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=36, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv1d(in_channels=36, out_channels=36,kernel_size=3, stride=2, padding=2)
        self.fc1 = nn.Linear(in_features=1800, out_features=LATENT)
        self.fc2 = nn.Linear(in_features=LATENT, out_features=self.n_outputs)
        self.LRelu = nn.LeakyReLU()


    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = out.view(out.size(0),-1)
        out = self.LRelu(self.fc1(out))
        out = self.fc2(out)
        return out
    
    def encode(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0),-1)
        x = self.LRelu(self.fc1(x))
        return x

    def act(self, state):
        n_obs = self.scaler.transform(state.reshape(1, -1))
        obs = torch.from_numpy(n_obs).float() #.unsqueeze(0) # [1,196]
        obs = obs.unsqueeze(1)
        a = self.forward(obs).squeeze().detach().cpu().numpy()
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


# class VAE(nn.Module):
    
#     def __init__(self):
#         super().__init__()

#         self.name = 'VAE'
#         self.LATENT = LATENT
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
#         # encoder
#         self.conv1 = nn.Conv1d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1)
#         self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1)
#         # self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)
#         self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
#         # z
#         self.mu = nn.Linear(1024, LATENT)
#         self.logvar = nn.Linear(1024, LATENT)
        
#         # decoder
#         self.fc = nn.Linear(LATENT, 6 * 6 * 256)  # Convert 1024 elements back to 4x4x64 tensor
#         self.dec_conv1 = nn.ConvTranspose1d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
#         self.dec_conv2 = nn.ConvTranspose1d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
#         self.dec_conv3 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
#         self.dec_conv4 = nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1)
        
#     def forward(self, x):
#         mu, logvar = self.encode(x)
#         z = self.latent(mu, logvar)
#         out = self.decode(z)
#         return out, mu, logvar   
    
#     def get_latent(self, x):
#         mu, logvar = self.encode(x)
#         z = self.latent(mu, logvar)
#         return z
        
#     def encode(self, x):
#         self.batch_size = 1
#         if len(x.shape)>3:
#             self.batch_size = x.shape[0]
        
#         out = F.relu(self.conv1(x))
#         out = F.relu(self.conv2(out))
#         # out = F.relu(self.conv3(out))
#         out = F.relu(self.adaptive_pool(out))
#         out = out.reshape(self.batch_size,1024)

#         mu = self.mu(out)
#         logvar = self.logvar(out)
#         return mu, logvar
        
#     def decode(self, z):
#         out = self.fc(z)
#         out = out.view(self.batch_size, 256, 6, 6)
    
#         out = F.relu(self.dec_conv1(out))
#         out = F.relu(self.dec_conv2(out))
#         out = torch.sigmoid(self.dec_conv3(out))
#         # out = F.relu(self.dec_conv3(out))
#         out = torch.sigmoid(self.dec_conv4(out))
#         return out
           
#     def latent(self, mu, logvar):
#         sigma = torch.exp(0.5*logvar)
#         eps = torch.randn_like(logvar).to(self.device)
#         z = mu + eps*sigma
#         return z
    
#     def load_parameters(self, src, version):
#       src = src+self.name.lower()+'-'+str(version)+'.pt'
#       if exists(src):
#           print("Loading model "+self.name.lower()+'-'+str(version)+" state parameters")
#           print("From :{}".format(src))
#           self.load_state_dict(torch.load(src, map_location=self.device))
#           return self
#       else:
#           print("Error no model "+self.name.lower()+'-'+str(version)+'.pt'+" found!")
#           exit(1)

    
#     def save_parameters(self, dest, version):
#       save_name = self.name.lower()+'-'+str(version)+'.pt'

#       if not exists(dest): 
#         mkdir(dest)
#       else: 
#           if exists(dest+save_name):
#             rename(dest+save_name, dest+save_name+'.bk')

#       torch.save(self.state_dict(), dest+save_name)


