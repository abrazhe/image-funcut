## some routines for filtering
from __future__ import division
import numpy as np
from scipy import signal
from scipy import ndimage

import itertools as itt


def gauss_kern(xsize=1.5, ysize=None):
    """
    Return a normalized 2D gauss kernel for convolutions

    Parameters:
      - `xsize`: :math:`\sigma_x`, standard deviation for x dimension
      - `ysize`: :math:`\sigma_y`, standard deviation for y dimension

    Returns:
      - `g` : the 2D kernel as an array
    """
    #norm = lambda _u: 2*int(np.ceil(_u))
    def norm(_u): return 2*int(np.ceil(_u))
    ysize = ysize and ysize or xsize
    xn,yn = norm(xsize), norm(ysize)
    x, y = np.mgrid[-xn:xn+1, -yn:yn+1]
    g = np.exp(-( 0.5*(x/float(xsize))**2 + 0.5*(y/float(ysize))**2))
    return g / g.sum()

def gauss_kern1d(size=1.5):
    """Given variance `size`, return 1D kernel
    """
    xsize = int(np.round(size))
    x = np.mgrid[-xsize:xsize+1]
    g = np.exp(-(x**2/size))
    return g/g.sum()

def gauss_blur(X,size=1.0):
    '''Return 2D Gauss blurred array `X` with a kernel of size `size`
    '''
    return signal.convolve2d(X,gauss_kern(size),'same')


def gauss_smooth(sig, sigma=1., dt = 1.0, order=0):
    """Perform Gauss smoothing (blurring) on signal `sig`

    Parameters:
      - `sig`:  an N-dimensional signal (vector, matrix, ...)
      - `sigma`: standard deviation of the Gauss filter
      - `dt`: sampling coefficient
      - `order`: order of the Gaussian function

    Returns:
      - blurred copy of `sig`

    See also:
      Uses functions scipy.ndimage.gaussian_filter and
      scipy.ndimage.gaussian_filter1d
    
    """
    sigma = sigma/dt
    ndim = np.ndim(sig)
    if ndim == 1:
	fn = ndimage.gaussian_filter1d
    else:
	fn = ndimage.gaussian_filter
    return fn(sig, sigma, order=order)


def mavg_DFoF(v, tau=90., dt=1.):
    """ Normalize signal `v` as :math:`(v-v_{baseline})/v_{baseline}` with smoothing

    Parameters:
      - `v`: input signal
      - `tau`: characteristic time of the smoothing function
      - `dt`: sampling interval

    Returns:
      - v/smooth(v) - 1 
    
    """
    baseline = gauss_smooth(v, tau, dt)
    zi = np.where(np.abs(baseline) < 1e-6)
    baseline[zi] = 1.0
    out = v/baseline - 1.0
    out[zi] = 0
    return out

def mavg_DFoSD(v, tau=90., dt=1.):
    """ Normalize signal `v` as standard score :math:`(v-v_{baseline})/\sigma_{v_{baseline}}` with smoothing

    Parameters:
      - `v`: input signal
      - `tau`: characteristic time of the smoothing function
      - `dt`: sampling interval

    Returns: Standard score
      - (v-smooth(v))/S.D.(smooth(v))
    
    """
    baseline = gauss_smooth(v, tau, dt)
    vd = v - baseline
    sd = np.std(vd)
    if sd < 1e-6:
	return np.zeros(vd.shape)
    return vd/sd


def _mirrorpd(k, L):
    if 0 <= k < L : return k
    else: return -(k+1)%L


def bspline_smooth(sig, phi = np.array([1./16, 1./4, 3./8, 1./4, 1./16])):
    """
    Smooth signal `sig` by 1D convolution with a cubic b-spline

    see `imfun.atrous.smooth` and `imfun.atrous.wavelet_denoise` for more variants
    """
    L = len(sig) 
    padlen = len(phi)
    assert L > padlen
    indices = map(lambda i: _mirrorpd(i, L),
                  range(-padlen, 0) + range(0,L) + range(L, L+padlen))
    padded_sig = sig[indices]
    apprx = np.convolve(padded_sig, phi, mode='same')[padlen:padlen+L]
    return apprx



# TODO: make it faster
def adaptive_medianf(arr, k = 2):
    """
    Perform adaptive median filtering on 2D array `arr`, by setting
    pixels to 3x3 local median if their value exceeds `k` times standard deviation
    over 3x3 neighborhood.

    TODO: convert to Ndimensions, and make it run faster
    """
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
    
def adaptive_medianf2(arr, k=3, s=1):
    import atrous
    sh = arr.shape
    approx = ndimage.median_filter(arr, 2*s+1)
    ns = atrous.estimate_sigma_mad(arr)
    d = arr-approx
    d[np.abs(d) > k*ns] = 0 
    return d + approx
    

def kalman_stack_filter(frames, seed='mean', gain=0.5, var=0.05, fn=lambda f:f):
    """Kalman stack filter similar to that of Imagej

    Input:
    ------
      - frames: array_like, e.g. nframes x nrows x ncolumns array or list of 2D
                images
      - seed: {'mean' | 'first' | 2D array}
              the seed to start with, defaults to the time-average of all
              frames, if 'first', then the first frame is used, if 2D array,
              this array is used as the seed
      - gain: overall filter gain
      - var: estimated environment noise variance

    Output:
    -------
      - new frames, an nframes x nrows x ncolumns array with filtered frames
    
    """
    if seed is 'mean' or None:
        seed = np.mean(frames,axis=0)
    elif seed is 'first':
        seed = frames[0]
    out = np.zeros_like(frames)
    predicted = fn(seed)
    Ek = var*np.ones_like(frames[0])
    var = Ek
    for k,M in enumerate(frames):
        Kk = 1.+ gain*(Ek/(Ek + var)-1)
        corrected = predicted + Kk*(M-predicted)
        err = (corrected-predicted)**2/predicted.max()**2 # unclear
        Ek = Ek*(1.-Kk) + err
        out[k] = corrected
        predicted = fn(out[k])
    return out
    

_bclose = ndimage.binary_closing
_bopen = ndimage.binary_opening
def opening_of_closing(a):
    "Return binary opening of binary closing of an array"
    return _bopen(_bclose(a))



from scipy import sparse
from scipy.sparse.linalg import spsolve

def baseline_als(y, lam, p, niter=10):
    """Implements an Asymmetric Least Squares Smoothing
    baseline correction algorithm
    (P. Eilers, H. Boelens 2005)
    """
    L = len(y)
    D = sparse.csc_matrix(np.diff(np.eye(L),2))
    w = np.ones(L)
    for i in xrange(niter):
	W = sparse.spdiags(w, 0, L, L)
	Z = W + lam*np.dot(D,D.T)
	z = spsolve(Z,w*y)
	w = p*(y>z) + (1-p)*(y<z)
    return z


# ---------------------------------------------


def test_kalman_stack_filter():
    "just tests if kalman_stack_filter function runs"
    print "Testing Kalman stack filter"
    import numpy as np
    try:
        test_arr = np.random.randn(100,64,64)
        arr2 = kalman_stack_filter(test_arr,)
        del  arr2, test_arr
    except Exception as e :
        print "filt.py: Failed to run kalman_stack_filter fuction"
        print "Reason:", e
