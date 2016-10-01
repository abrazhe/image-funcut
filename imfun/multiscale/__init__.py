"""
## Mulstiscale decomposition and reconstruction routines
"""

import numpy as np
from scipy import ndimage

from . import atrous
from . import mmt
from . import emd
from . import mvm
from . import utils
from .utils import pyramid_from_atrous
from .utils import pyramid_from_zoom

_dtype_ = np.float32


_boundary_mode = 'nearest'

sigmaej_starlet = [[0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],   # 0D
                   [7.235e-01, 2.854e-01, 1.779e-01, 1.222e-01, 8.581e-02, 6.057e-02,  4.280e-02, 3.025e-02, 2.138e-02, 1.511e-02, 1.067e-02, 7.512e-03], #1D
                   [0.890, 0.201, 0.086, 0.042, 0.021, 0.010, 0.005],   # 2D
                   [0.956, 0.120, 0.035, 0.012, 0.004, 0.001, 0.0005]]  # 3D




## Default spline wavelet scaling function
_phi_ = np.array([1./16, 1./4, 3./8, 1./4, 1./16], _dtype_)

