from pkg_resources import get_distribution, DistributionNotFound

__project__ = 'image-funcut'
__version__ = None  # required for initial installation

try:
    __version__ = get_distribution(__project__).version
except DistributionNotFound:
    VERSION = __project__ + '-' + '(local)'
else:
    VERSION = __version__

import lib
import fseq
import fnmap
import ui
import synthdata
import cluster
import pica
import atrous
import mmt
import multiscale
import mvm
import fnutils
