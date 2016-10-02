from pkg_resources import get_distribution, DistributionNotFound

__project__ = 'image-funcut'
__version__ = None  # required for initial installation

try:
    __version__ = get_distribution(__project__).version
except DistributionNotFound:
    VERSION = __project__ + '-' + '(local)'
else:
    VERSION = __version__

from . import bwmorph
from . import cluster, components
from . import external
from . import fnmap, fnutils, fseq
from . import io
from . import lib
from . import multiscale
from . import ofreg
from . import track, ui
from . import ui
