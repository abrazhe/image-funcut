from pkg_resources import get_distribution, DistributionNotFound

__project__ = 'image-funcut'
__version__ = None  # required for initial installation

try:
    __version__ = get_distribution(__project__).version
except DistributionNotFound:
    VERSION = __project__ + '-' + '(local)'
else:
    VERSION = __version__

from . import atrous, bwmorph, cluster, fseq, fnmap, fnutils, lib
from . import ioraw
from . import mmt, multiscale, mvm
from . import opflowreg, pica, track, ui
from . import external

## import lib
## import fseq
## import fnmap
## import ui
## import synthdata
## import cluster
## import pica
## import atrous
## import mmt
## import multiscale
## import mvm
## import fnutils
## import bwmorph
## import track
## import opflowreg
