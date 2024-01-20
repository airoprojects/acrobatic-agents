#! /bin/bash

mujoco_path=$HOME'/miniconda3/envs/test/lib/python3.8/site-packages/gymnasium/envs/mujoco/'

# saving a backup of current humanoid env in mujoco library
mv $mujoco_path'humanoid_v4.py' $mujoco_path'humanoid_v4.bk.py' 

# copining new (custom) humanoid in mujoco library
cp ./humanoid_v4.py  $mujoco_path'humanoid_v4.py'

# coping utils functions and motion capture data inside mujoco library
cp -r ./utils/ $mujoco_path