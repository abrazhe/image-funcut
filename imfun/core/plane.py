
import numpy as np
from scipy import optimize as opt

def plane(pars, x,y):
	kx,ky,z = pars
        return x*kx + y*ky + z

def remove_plane(arr, pars):
	shape = arr.shape
	X,Y = np.meshgrid(*map(range,shape[::-1]))
	return arr - plane(pars, X, Y)

def fit_plane(arr):
    def _plane_resid(pars, Z, shape):
        Z = np.reshape(Z,shape)
        X,Y = np.meshgrid(*map(range,shape[::-1]))
        return (Z - plane(pars,X,Y)).flatten()
    p0 = np.random.randn(3)
    p1 = opt.leastsq(_plane_resid, p0, (arr.flatten(), arr.shape))[0]
    return p1
