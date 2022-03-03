REGISTRY = {}

from .thgc_agent import THGCAgent
from .rnn_agent import RNNAgent
REGISTRY["thgc"] = THGCAgent
REGISTRY["rnn"] = RNNAgent
