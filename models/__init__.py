import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "lib"))

#tipo java
from .bco_fc import BCOAgentFC
from .bco_cnn import BCOCNN