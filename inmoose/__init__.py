import importlib.metadata

__version__ = importlib.metadata.version(__package__)

# force RTLD_GLOBAL when loading common_cpp
from sys import getdlopenflags, setdlopenflags
from os import RTLD_GLOBAL

default_dlopen_flags = getdlopenflags()
setdlopenflags(default_dlopen_flags | RTLD_GLOBAL)
from . import common_cpp

setdlopenflags(default_dlopen_flags)
del default_dlopen_flags

from . import edgepy
from . import pycombat
from . import utils
from . import consensus_clustering

import logging

logging.basicConfig(level=logging.INFO, format="[{levelname}] {message}", style="{")
logging.captureWarnings(True)
