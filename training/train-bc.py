import torch
from torch.utils.data.dataset import Dataset, random_split

import os
import sys
import argparse
from os import path
from git import Repo
from tqdm import tqdm

class ExpertDataSet(Dataset):

  def __init__(self, expert_observations, expert_actions):
    self.observations = expert_observations
    self.actions = expert_actions

  def __getitem__(self, index):
    return (self.observations[index], self.actions[index])

  def __len__(self):
    return len(self.observations)
    

class ExpertDataLoader():

  def __init__(self, expert_observations, expert_actions):
    self.expert_dataset = ExpertDataSet(expert_observations, expert_actions)
    self.train_loader = None
    self.test_loader = None

  def __call__(self, batch_size=64, train_prop=0.8):
    train_size = int(train_prop * len(self.expert_dataset))
    test_size = len(self.expert_dataset) - train_size

    train_expert_dataset, test_expert_dataset = random_split(self.expert_dataset, [train_size, test_size])

    self.train_loader = torch.utils.data.DataLoader(dataset=train_expert_dataset, batch_size=batch_size, shuffle=True)
    self.test_loader = torch.utils.data.DataLoader(  dataset=test_expert_dataset, batch_size=batch_size, shuffle=True)

    return self.train_loader, self.test_loader




  


    