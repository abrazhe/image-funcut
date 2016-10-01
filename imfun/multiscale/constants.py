import numpy as np

_boundary_mode = 'nearest'
_dtype_ = np.float32
_phi_ = np.array([1./16, 1./4, 3./8, 1./4, 1./16], _dtype_)


__all__=['_boundary_mode',
         '_dtype_',
         '_phi_']

