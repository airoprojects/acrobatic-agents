# Acrobatic Agents
RL - Sapienza (AIRO) 2023/2024 -  Bruno Francesco Nocera, Leonardo Colosi

---
## Table of content
0. [Introduction](#introduction)
1. [Getting Started](#getting-started)
5. [Resources](#useful-resources)
6. [References](#references)
---

# Introduction
The goal of this project was to train a humanoid agent to perform some acrobatics motions via Imitation Learning. The environment in which the simulation happens is the Humanoid3d world from pybullet [[2]](#references). In order to train our agent we choose the PPO-Agent, also implemented by [[2]](#references), as the expert to imitate. The results shows that, with enough data from the expert, even a simple network is able to learn task such as performing a back-flip or a spin-kick. More interesting is the fact that combining different observation kind from the expert in a single training it is possible to develop the ability to perform different tasks with the same network.


# Getting started

## Environment Setup 
```code
conda create --name hum_rl python=3.7
```

***IMPORTANT***: Install pytorch from the official [website](https://pytorch.org/get-started/locally/). Make sure to select the right version according to your system characteristics!

## Installation of gym from the official repo
```code
git clone https://github.com/openai/gym.git ~/git || (cd ~/git ; git pull)
pip install -e ~/git
```


## Some quick fixes may be needed

#### To solve MESA related problems on ubuntu 22.04 : 
```code
conda install -c conda-forge libstdcxx-ng
```

#### To solve mpi4py
```code
conda install -c conda-forge mpi4py mpich
```

#### A downgrade of protobuf is required 
```code
pip install protobuf==3.20.*
```

## Install all other requirements

``` code
pip install -r requirements.txt --no-cache-dir
``` 

## Library incompatibility

#### Error with new gym registry:
```code 
File "train.py", line 318, in <module>
   main()
 File "train.py", line 200, in main
   frame_skip=args.action_repeat
 File "/home/anavani/anaconda3/envs/rad/lib/python3.7/site-packages/dmc2gym/__init__.py", line 28, in make
   if not env_id in gym.envs.registry.env_specs:
AttributeError: 'dict' object has no attribute 'env_specs'
```
Solution:

`gym.envs.registry` was previously a complex class that we replaced with a dictionary for simplicity.
The code should just need to be changed to if env_id not in `gym.envs.registry`

# Useful resources
To explore some resources used/related to the project look here [resources](resources)

# References
[1] Peng, Xue Bin and Abbeel, Pieter and Levine, Sergey and van de Panne, Michiel, 2018, [article](https://arxiv.org/pdf/1804.02717.pdf)

[2] Bullet Physics SDK  [bullet3](https://github.com/bulletphysics/bullet3.git)

Backflip demonstration:

https://drive.google.com/file/d/1oAatlrRgl3KW3v4MTVYScH36jAs1xo3_/view?usp=drive_link
