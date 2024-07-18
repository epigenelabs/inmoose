import importlib.metadata

__version__ = importlib.metadata.version(__package__)

# force RTLD_GLOBAL when loading common_cpp
from sys import getdlopenflags, setdlopenflags
from os import RTLD_GLOBAL

default_dlopen_flags = getdlopenflags()
setdlopenflags(default_dlopen_flags | RTLD_GLOBAL)
from . import common_cpp as common_cpp

setdlopenflags(default_dlopen_flags)
del default_dlopen_flags

from . import edgepy as edgepy
from . import pycombat as pycombat
from . import utils as utils
from . import consensus_clustering as consensus_clustering
from . import deseq2 as deseq2
from . import diffexp as diffexp
