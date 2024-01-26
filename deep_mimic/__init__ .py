'''
Note: this module contains code from bullet3 library: (https://github.com/bulletphysics/bullet3/tree/master)
It is a slight customization of the Deep Mimic Env in order to make it works with a IL Agent implemented in pytorch
'''

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "lib"))

from .rl_util import update_world
from .rl_util import build_world
