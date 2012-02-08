## some routines for filtering
from __future__ import division
import numpy as np
from scipy import signal
from scipy import ndimage

import itertools as itt


def gauss_kern(xsize=1.5, ysize=None):
    """ Returns a normalized 2D gauss kernel for convolutions """
    xsize = int(xsize)
    ysize = ysize and int(ysize) or xsize
    x, y = np.mgrid[-xsize:xsize+1, -ysize:ysize+1]
    g = np.exp(-(x**2/float(xsize) + y**2/float(ysize)))
    return g / g.sum()


def gauss_blur(X,size=1.0):
    return signal.convolve2d(X,gauss_kern(size),'same')


def gauss_smooth(sig, sigma=1., dt = 1.0, order=0):
    sigma = sigma/dt
    ndim = np.ndim(sig)
    if ndim == 1:
	fn = ndimage.gaussian_filter1d
    else:
	fn = ndimage.gaussian_filter
    return fn(sig, sigma, order=order)


def mavg_DFoF(v, tau=90., dt=1.):
    baseline = gauss_smooth(v, tau, dt)
    zi = np.where(np.abs(baseline) < 1e-6)
    baseline[zi] = 1.0
    out = v/baseline - 1.0
    out[zi] = 0
    return out

def mavg_DFoSD(v, tau=90., dt=1.):
    baseline = gauss_smooth(v, tau, dt)
    vd = v - baseline
    sd = np.std(vd)
    if sd < 1e-6:
	return np.zeros(vd.shape)
    return vd/sd


def mirrorpd(k, L):
    if 0 <= k < L : return k
    else: return -(k+1)%L


def bspline_denoise(sig, phi = np.array([1./16, 1./4, 3./8, 1./4, 1./16])):
    L = len(sig) 
    padlen = len(phi)
    assert L > padlen
    indices = map(lambda i: mirrorpd(i, L),
                  range(-padlen, 0) + range(0,L) + range(L, L+padlen))
    padded_sig = sig[indices]
    apprx = np.convolve(padded_sig, phi, mode='same')[padlen:padlen+L]
    return apprx



# TODO: make it faster
def adaptive_medianf(arr, k = 2):
    sh = arr.shape
    out = arr.copy()
    for row in xrange(1,sh[0]-1):
        for col in xrange(1,sh[1]-1):
            sl = (slice(row-1,row+2), slice(col-1,col+2))
            m = np.mean(arr[sl])
            sd = np.std(arr[sl])
            if (arr[row,col] > m+k*sd) or \
                   (arr[row,col] < m- k*sd):
                out[row, col] = np.median(arr[sl])
    return out
    


def opening_of_closing(a):
    "performs binary opening of binary closing of an array"
    bclose = ndimage.binary_closing
    bopen = ndimage.binary_opening
    return bopen(bclose(a))





