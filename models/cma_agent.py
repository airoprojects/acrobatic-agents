import cma

import sys
import time
import pickle
import joblib
import argparse
from git import Repo
from tqdm import tqdm

from os.path import join, exists
from os import mkdir, unlink, listdir, getpid, remove, rename

import torch
import numpy as np 

from models import BCOAgentFC
from models import BCO_cnn
from models import Controller
import deep_mimic.rl_util as dm

# set up train device
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'
# print("device: {}".format(device))

# # setup project root dir
# repo = Repo(".", search_parent_directories=True)
# root_dir = repo.git.rev_parse("--show-toplevel")
# print("root: {}".format(root_dir))

LATENT = 100

import torch
import torch.nn as nn
from torchvision import transforms
torch.manual_seed(42)

class CMAPolicy(nn.Module):

    def __init__(self, 
                 encoder_version=1,
                 controller_version=1,
                 scaler_version = 6000,
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        super(CMAPolicy, self).__init__()
        

        # global variables
        self.update_timestep = 1. / 240.
        self.device = device
        # animating = True
        # step = False
        args = sys.argv[1:]

        # setup project root dir
        repo = Repo(".", search_parent_directories=True)
        self.root_dir = repo.git.rev_parse("--show-toplevel")
        print("root: {}".format(self.root_dir))

        # env
        self.world = dm.build_world(args, True, enable_stable_pd=False)
        self.obs_dim = self.world.env.get_state_size()
        self.action_dim = self.world.env.get_action_size()
        
        # models
        self.e_version = encoder_version
        self.encoder = BCO_cnn(self.obs_dim, self.action_dim).to(device).eval()
        self.c_version = controller_version
        self.c = Controller(LATENT, self.action_dim).to(device)
        self.override = self

        # scaler
        scaler_path = self.root_dir+'/data/scaler-'+str(scaler_version)+'.joblib'
        self.scaler = joblib.load(scaler_path)  

        # cma training parameters
        self.sigma = 0.2
        self.pop_size = 6
        self.n_samples = 3
        self.fixed_seed = 588039 
        self.max_reward = 1000
        # self.stop_condiction = 700 # stop at (1000 - reward) e.g. s.c. = 200 --> reward = 800
        self.target_mean = 200 # target mean reward
  
    def act(self, state):
        
      n_obs = self.scaler.transform(state.reshape(1, -1))
      obs = torch.from_numpy(n_obs).float() #.unsqueeze(0) # [1,196]
      
      if self.encoder.name:
            obs = obs.unsqueeze(1)

      # latent representation
      z = self.encoder(obs).squeeze().detach().cpu().numpy()

      # action
      a = self.c(z)

      return a


    def train(self):
        """Train the entire network or just the controller module"""

        # Train controller
        print("Attempting to load previous best...")
       
        cur_best = 100000000000 # max cap
        cur_mean = -100000000000 # min cap

        src =  self.root_dir+'/checkponts/'
        file_name = self.c.name.lower()+'-'+str(self.c_version)+'.pt'
        if exists(src + file_name): 
            self.c = self.c.load_parameters(src, self.c_version)
            print("Previous controller loaded")
            # cur_best = self.c.load(self.modules_dir, get_value=True)
            # print("Best current value for the objective function: {}".format(cur_best))

        # set up cma parameters
        params = self.c.parameters()
        flat_params = torch.cat([p.detach().view(-1) for p in params], dim=0).cpu().numpy()
        es = cma.CMAEvolutionStrategy(flat_params, self.sigma, {'popsize':self.pop_size}) #'seed':self.fixed_seed

        # log variables for cma controller
        display = True
        generation = 0

        print("Starting CMA training")
        print("Generation {}".format(generation+1))

        while not es.stop(): 

            if cur_mean >= self.target_mean:
            # if cur_best <= self.stop_condiction:
                print("Already better than the target value")
                print("Stop training...")
                break

            # compute solutions
            result_list = [0] * self.pop_size  
            solutions = es.ask()

            if display: pbar = tqdm(total=self.pop_size*self.n_samples)
            
            for s_id, params in enumerate(solutions):
                for _ in range(self.n_samples):
                    value, _ = self.cma_rollout( 
                                                self,
                                                self.world,
                                                params, 
                                                device=self.device,
                                            )
                    result_list[s_id] += value / self.n_samples
                    if display: pbar.update(1)

            if display: pbar.close()

            # cma step
            es.tell(solutions, result_list)
            es.disp()

            # evaluation and saving
            print("Evaluating...")
            best_params, best, cur_mean = self.evaluate(solutions, result_list, run=6)

            print("Current evaluation of the objactive function (J): {} \nNote: this value should decrease".format((best))) 
            print("Current mean reward: {}".format(cur_mean)) 

            if not cur_best or cur_best > best: 
                print("Previous best with value J = {}...".format((cur_best)))
                cur_best = best
                print("Saving new best with value J = {}...".format((cur_best)))
    
                # load parameters into controller
                unflat_best_params = self.unflatten_parameters(best_params, self.c.parameters(), self.device)
                for p, p_0 in zip(self.c.parameters(), unflat_best_params):
                    p.data.copy_(p_0)

                # saving
                self.c.save_parameters(self.root_dir+'/checkpoints/', self.c_version)
                # self.save()

                print("Rendering...")
                self.evaluate(solutions, result_list, render=True, run=3)

            if cur_mean >= self.target_mean:
                print("Terminating controller training with value {}...".format(-cur_best))
                break

            generation += 1
            print("Generation {}".format(generation+1))

        return
    
    def evaluate(self, solutions, results, render=False, run=6):
        index_min = np.argmin(results)
        best_guess = solutions[index_min]
        best_estimates = []
        reward_estimate = []

        p_list = []
        for s_id in range(run):
            p_list.append((s_id, best_guess))

        for _ in tqdm(range(run)):
            value, reward = self.cma_rollout(
                                            self,
                                            self.world,
                                            best_guess, 
                                            device=self.device, 
                                        )    
            best_estimates.append(value)
            reward_estimate.append(reward)
        return best_guess, np.mean(best_estimates), np.mean(reward_estimate)
    

    def cma_rollout(self, world, params, device):
        steps = 0
        total_reward = 0
        max_reward = 500

        if params is not None:
            params = self.unflatten_parameters(params, self.c.parameters(), device)

        # load parameters into agent controller
        for p, p_0 in zip(self.c.parameters(), params):
            p.data.copy_(p_0)

        while True:

            timeStep = self.update_timestep
            s, a = world.update(timeStep, override=self)

            reward = world.env.calc_reward(agent_id=0)
            total_reward += reward
            steps+=1

            end_episode = world.env.is_episode_end()
            if (end_episode or steps>= 1000):
                print("total_reward=",total_reward)
                total_reward=0
                steps = 0
                world.end_episode()
                world.reset()
                break

        return (max_reward - total_reward), total_reward
    

    def unflatten_parameters(self, params, example, device):
        """ Unflatten parameters. Note: example is generator of parameters (module.parameters()), used to reshape params """
        
        params = torch.Tensor(params).to(device)
        idx = 0
        unflattened = []
        for e_p in example:
            unflattened += [params[idx:idx + e_p.numel()].view(e_p.size())]
            idx += e_p.numel()
        return unflattened
                    
        
#############################################################################
    
    def save(self):
        print("Saving model")
        torch.save(self.state_dict(), 'model.pt')

    def load(self): 
        self.load_state_dict(torch.load('model.pt', map_location=self.device))

    def to(self, device):
        ret = super().to(device)
        ret.device = device
        return ret

    
