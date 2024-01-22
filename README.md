# Acrobatic Agents
RL - Sapienza (AIRO) 2023/2024 -  Bruno Francesco Nocera, Leonardo Colosi


## Environment Setup 
```code
conda create --name hum_rl python=3.7
```

Install pytorch from the official [website](https://pytorch.org/get-started/locally/). Make sure to select the right version according to your system characteristics!

Install old version of gym (?)
```code
git clone https://github.com/openai/gym.git
cd gym
pip install -e .
```

Install other requirements

``` code
pip install -r requirements.txt --no-cache-dir
``` 

## Some quick fixes

#### To solve MESA related problems on ubuntu 22.04 : 
```code
conda install -c conda-forge libstdcxx-ng
```

###
conda install -c conda-forge mpi4py mpich
###


#### protobuf downgrade needed
```
pip install protobuf==3.20.*
```

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

## Useful resources
To explore some resources used/related to the project look here [resources](resources)


