from metabci.brainda.algorithms.deep_learning import EEGNet, ShallowNet, ConvCA
from metabci.brainda.algorithms.deep_learning.deepnet import Deep4Net as DeepNet

from .braindecode.eegnet import EEGNetv4
from .braindecode.deep4 import Deep4Net
from .braindecode.shallow_fbcsp import ShallowFBCSPNet

from .fbcnet import FBCNet
from .cspnet.model import Tensor_CSPNet, Graph_CSPNet
from .lmda_net import LMDANet

__all__ = [ 
    'EEGNet',
    'ShallowNet',
    'DeepNet',
    'ConvCA',
    'Formatdata',
    'EEGNetv4',
    'Deep4Net',
    'ShallowFBCSPNet',
    'FBCNet',
    'Tensor_CSPNet',
    'Graph_CSPNet',
    'LMDANet',
    'DL_Classifier'
]
