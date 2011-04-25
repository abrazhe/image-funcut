## some routines for filtering
from __future__ import division
import numpy as np
from scipy import signal

def gauss_kern(xsize=1.5, ysize=None):
    """ Returns a normalized 2D gauss kernel for convolutions """
    xsize = int(xsize)
    ysize = ysize and int(ysize) or xsize
    x, y = np.mgrid[-xsize:xsize+1, -ysize:ysize+1]
    g = np.exp(-(x**2/float(xsize) + y**2/float(ysize)))
    return g / g.sum()


def gauss_blur(X,size=1.0):
    return signal.convolve2d(X,gauss_kern(size),'same')

def in_range(low, high):
    return lambda x: (x >=low)*(x < high)

sigmaej = [[0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],   # 0D 
           [0.700, 0.323, 0.210, 0.141, 0.099, 0.071, 0.054],   # 1D
           [0.889, 0.200, 0.086, 0.041, 0.020, 0.010, 0.005],   # 2D
           [0.956, 0.120, 0.035, 0.012, 0.004, 0.001, 0.0005]] # 3D

# TODO: make it faster
def adaptive_medianf(arr, k = 2):
    sh = arr.shape
    out = arr.copy()
    for row in xrange(1,sh[0]-1):
        for col in xrange(1,sh[1]-1):
            sl = (slice(row-1,row+1), slice(col-1,col+1))
            m = np.mean(arr[sl])
            sd = np.std(arr[sl])
            if (arr[row,col] > m+k*sd) or \
                   (arr[row,col] < m- k*sd):
                out[row, col] = np.median(arr[sl])
    return out
    


def dec_atrous2d(arr2d, lev, kern=None, boundary='symm'):
    """
    Do 2d a'trous wavelet transform with B3-spline scaling function

    This is a convolution version, where kernel is zero-upsampled
    explicitly. Not fast.

    Inputs:
    ---------
    arr2d : 2D array
    kern  : low-pass filter kernel (B3-spline by default)
    boundary : boundary conditions (passed to scipy.signal.convolve2d, 'symm'
               by default)
    Outputs:
    ---------
    list of wavelet details + last approximation
    
    """
    _b3spline1d = np.array([1./16, 1./4, 3./8, 1./4, 1./16])
    __x = _b3spline1d.reshape(1,-1)
    _b3spl2d = np.dot(__x.T,__x)
    if kern is None: kern = _b3spl2d
    if lev <= 0: return arr2d
    shapecheck = map(lambda a,b:a>b, arr2d.shape, kern.shape)
    assert np.all(shapecheck)
    # approximation:
    approx = signal.convolve2d(arr2d, kern, mode='same',
                               boundary=boundary)  
    w = arr2d - approx   # wavelet details
    upkern = zupsample(kern)
    shapecheck = map(lambda a,b:a>b, arr2d.shape, upkern.shape)
    if lev == 1:
        return [w, approx]
    elif not np.all(shapecheck):
        print "Maximum possible decomposition level reached, not advancing any more"
        return [w, approx]
    else:
        return [w] + dec_atrous2d(approx,lev-1,upkern,boundary) 

def zupsample(arr):
    "Upsample array by interleaving it with zero values"
    sh = arr.shape
    newsh = [d*2 for d in sh]
    o = np.zeros(newsh,dtype=arr.dtype)
    o[[slice(None,None,2) for d in sh]] = arr
    return o


def rec_atrous2d(coefs, level=None):
    "Reconstruct from atruos decomposition. Last coef is last approx"
    return np.sum(coefs[-1:level:-1], axis=0)

def represent_support(supp):
    out = [2**(j+1)*supp[j] for j in range(len(supp)-1)]
    return np.sum(out, axis=0)

def get_support(coefs, th, neg=False):
    out = []
    fn = neg and np.less or np.greater
    for j,w in enumerate(coefs[:-1]):
        t  = np.iterable(th) and th[j] or th
        out.append(fn(np.abs(w), t*sigmaej[2][j]))
    out.append(np.ones(coefs[-1].shape)*(not neg))
    return out

def invert_mask(m):
    def _neg(a):
        return not a
    return np.vectorize(_neg)(m)

def arr_or(a1,a2):
    return np.vectorize(lambda x,y: x or y)(a1,a2)

def estimate_sigma(arr, coefs, k=3, eps=0.01, max_iter=1e9):
    sprev = estimate_sigma_mad(coefs[0])
    #sprev = arr.std()
    for j in xrange(int(max_iter)):
        supp = get_support(coefs, sprev*k, neg=True)
        mask = np.prod(supp[:-1], axis=0)
        snext =  np.std((arr-coefs[-1])[mask])
        #print snext, sprev
        assert np.sum(mask) > 0
        if abs(sprev-snext)/snext <= eps:
            return snext
        sprev = snext
    return sprev


def estimate_sigma_mad(coefarr):
    return np.median(np.abs(coefarr))/(0.6745*sigmaej[2][0])

def wavelet_enh_std(f, level=4, out = 'rec',absp = False):
    fw = dec_atrous2d(f, level)
    if absp:
        supp = map(lambda x: abs(x) > x.std(), fw)
    else:
        supp = map(lambda x: x > x.std(), fw)
    if out == 'rec':
        filtcoef = [x*w for x,w in zip(supp, fw)]
        return rec_atrous2d(filtcoef)
    elif out == 'supp':
        return represent_support(supp)
        
def wavelet_denoise(f, k=[3,3,2,2], level = 4, noise_std = None):
    if np.iterable(k):
        level = len(k)
    coefs = dec_atrous2d(f, level)
    if noise_std is None:
        noise_std = estimate_sigma(f, coefs) / 0.974 # magic value
    supp = get_support(coefs, np.array(k)*noise_std)
    filtcoef =  [c*s for c,s in zip(coefs, supp)]
    return rec_atrous2d(filtcoef)
