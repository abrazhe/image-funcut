## some routines for filtering
from __future__ import division
import numpy as np
from scipy import signal


def gauss_blur(X,size=1.0):
    return signal.convolve2d(X,gauss_kern(size),'same')



_b3spline1d = np.array([1./16, 1./4, 3./8, 1./4, 1./16])

__x = _b3spline1d.reshape(1,-1)
_b3spl2d = np.dot(__x.T,__x)


def atrous2d(arr, lev, boundary='symm'):
    approx = signal.convolve2d(arr,_b3spl2d,
                               mode='same',
                               boundary=boundary)  # approximation
    w = arr - approx                               # wavelet details
    if lev <= 0: return arr
    if lev == 1: return [w, approx]
    else:        return [w] + atrous1(approx,lev-1,boundary)
        
