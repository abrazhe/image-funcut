# -*- coding: utf-8 -*-
""" Functions for à trous wavelet transforms
Synonyms: stationary wavelet transform, non-decimated wavelet transform
"""
from __future__ import division

import numpy as np
from scipy import signal
from scipy import ndimage
from scipy.ndimage import convolve1d

import itertools as itt


_dtype_ = np.float32

## this is used for noise estimation and support calculation
## level =  1      2      3      4      5      6      7
sigmaej = [[0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],   # 0D 
           [0.700, 0.323, 0.210, 0.141, 0.099, 0.071, 0.054],   # 1D
           [0.890, 0.201, 0.086, 0.042, 0.021, 0.010, 0.005],   # 2D
           [0.956, 0.120, 0.035, 0.012, 0.004, 0.001, 0.0005]] # 3D

sigmaej.__doc__ = """A table for noise variance at different levels of
decomposition for data of different dimensionality"""

def mc_levels(size=(256,256),level=3, N = 1e3):
    import sys
    images = (randn(*size) for i in arange(N))
    out  = np.zeros((N,level))
    for n,im in enumerate(images):
	x = np.mean(out[:n], axis=0)
	#sys.stderr.write('\r image %06d out of %d, current: %05.3f'%(n+1,N, x))
	print n+1, 'current: ', x
	out[n] = map(np.std, decompose(im, level)[:-1])
    return np.mean(out, axis=0)
    	   

## Default spline wavelet scaling function
_phi_ = np.array([1./16, 1./4, 3./8, 1./4, 1./16], _dtype_)

def locations(shape):
    return itt.product(*map(xrange, shape))

def decompose(arr, *args, **kwargs):
    "Dispatcher on 1D, 2D or 3D data decomposition"
    ndim = arr.ndim
    if ndim == 1:
        decfn = decompose1d
    elif ndim == 2:
        decfn = decompose2d
    elif ndim == 3:
        decfn = decompose3d
    else:
        print "Can't work with %d dimensions yet, returning"%ndim
        return
    return decfn(arr, *args, **kwargs)

def decompose1d(sig, level, phi=_phi_):
    """1d stationary wavelet transform"""
    sig = sig.astype(_dtype_)
    apprx = convolve1d(sig, phi, mode='mirror')
    w = (sig - apprx) # wavelet coefs
    L = len(sig)
    if level <= 0: return sig
    elif level == 1 or L < len(zupsample(phi)): return [w, apprx]
    else: return [w] + decompose1d(apprx, level-1, zupsample(phi))

def decompose2d(arr2d, level, kern=None, boundary='symm'):
    """
    2d stationary wavelet transform with B3-spline scaling function

    This is a convolution version, where kernel is zero-upsampled
    explicitly. Not fast.

    Parameters:
      - arr2d : 2D array
      - level : level of decomposition
      - kern  : low-pass filter kernel (B3-spline by default)
      - boundary : boundary conditions (passed to scipy.signal.convolve2d, 'symm'
               by default)
    Returns:
      list of wavelet details + last approximation
    
    """
    _b3spline1d = np.array(_phi_, _dtype_)
    __x = _b3spline1d.reshape(1,-1)
    _b3spl2d = np.dot(__x.T,__x)
    if kern is None: kern = _b3spl2d
    if level <= 0: return arr2d
    shapecheck = map(lambda a,b:a>b, arr2d.shape, kern.shape)
    assert np.all(shapecheck)
    arr2d = arr2d.astype(_dtype_)
    # approximation:
    approx = signal.convolve2d(arr2d, kern, mode='same',
                               boundary=boundary)  
    w = arr2d - approx   # wavelet details
    upkern = zupsample(kern)
    shapecheck = map(lambda a,b:a>b, arr2d.shape, upkern.shape)
    if level == 1:
        return [w, approx]
    elif not np.all(shapecheck):
        print "Maximum allowed decomposition level reached, not advancing any more"
        return [w, approx]
    else:
        return [w] + decompose2d(approx,level-1,upkern,boundary) 

def f2d(phi):
    v = phi.reshape(1,-1)
    return np.dot(v.T,v)


def decompose2p5d(arr, level=1,
                  phi = _phi_,
                  boundary = 'mirror'):
    """Semi-separable a trous wavelet decomposition for 3D data"""
    phi2d = f2d(phi)
    if level <= 0: return arr
    arr = arr.astype(_dtype_)
    approx = np.zeros(arr.shape,_dtype_)
    for k in xrange(arr.shape[0]):
        approx[k] = signal.convolve2d(arr[k], phi2d, mode='same',
                                      boundary='symm')
    details = arr - approx
    upkern = zupsample(phi)
    shapecheck = map(lambda a,b:a>b, arr.shape, upkern.shape)
    if level == 1:
        return [details, approx]
    elif not np.all(shapecheck):
        print "Maximum allowed decomposition level reached, returning"
        return [details, approx]
    else:
        return [details] + decompose2p5d(approx, level-1, upkern)



def decompose3d(arr, level=1,
                  phi = _phi_,
                  boundary = 'mirror'):
    "Semi-separable \'a trous wavelet decomposition for 3D data"
    phi2d = f2d(phi)
    if level <= 0: return arr
    arr = arr.astype(_dtype_)
    tapprox = np.zeros(arr.shape,_dtype_)
    for loc in locations(arr.shape[1:]):
        v = arr[:,loc[0], loc[1]]
        tapprox[:,loc[0], loc[1]] = convolve1d(v, phi, mode=boundary)
    approx = np.zeros(arr.shape,_dtype_)
    for k in xrange(arr.shape[0]):
        approx[k] = signal.convolve2d(tapprox[k], phi2d, mode='same',
                                      boundary='symm')
    details = arr - approx
    upkern = zupsample(phi)
    shapecheck = map(lambda a,b:a>b, arr.shape, upkern.shape)
    if level == 1:
        return [details, approx]
    elif not np.all(shapecheck):
        print "Maximum allowed decomposition level reached, returning"
        return [details, approx]
    else:
        return [details] + decompose3d(approx, level-1, upkern)

def zupsample(arr):
    "Upsample array by interleaving it with zero values"
    sh = arr.shape
    newsh = [d*2-1 for d in sh]
    o = np.zeros(newsh,dtype=arr.dtype)
    o[[slice(None,None,2) for d in sh]] = arr
    return o


def rec_atrous(coefs, level=None):
    "Reconstruct from a trous decomposition. Last coef is last approx"
    return np.sum(coefs[-1:level:-1], axis=0)

def represent_support(supp):
    out = [2**(j+1)*supp[j] for j in range(len(supp)-1)]
    return np.sum(out, axis=0)

def get_support(coefs, th, neg=False, modulus = True,soft=False):
    out = []
    nd = len(coefs[0].shape)
    fn = neg and np.less or np.greater
    for j,w in enumerate(coefs[:-1]):
	sj= sigmaej[nd][j]

	if np.iterable(th): t = th[j]
	else: t = th

	if modulus: wa = np.abs(w)
	else: wa = w

	if np.iterable(t):
	    out.append((wa > t[0]*sj)*(wa<=t[1]*sj))
	else:
	    mask = fn(wa, t*sj)
	    if soft:
		out.append(1.0*mask*np.sign(w)*(np.abs(w)-t*sj))
	    else:
		out.append(mask)
    out.append(np.ones(coefs[-1].shape)*(not neg))
    return out


def estimate_sigma(arr, coefs, k=3, eps=0.01, max_iter=1e9):
    sprev = estimate_sigma_mad(coefs[0], True)
    for j in xrange(int(max_iter)):
        supp = get_support(coefs, sprev*k, neg=True)
        mask = np.prod(supp[:-1], axis=0)
        snext =  np.std((arr-coefs[-1])[mask])
        if abs(sprev-snext)/snext <= eps:
            return snext
        sprev = snext
    return sprev

def estimate_sigma_kclip(arr, k=3.0, max_iter=3):
    d = np.ravel(decompose(arr,1)[0])
    for j in xrange(max_iter):
	d = d[abs(d) < k*np.std(d)]
    return np.std(d)

def estimate_sigma_mad(arr, is_details = False):
    if is_details:
	w1 = arr
    else:
	w1 = decompose(arr,1)[0]
    nd = w1.ndim
    return np.median(np.abs(w1))/(0.6745*sigmaej[nd][0])

def smooth(arr, level=1):
    return decompose(arr, level)[-1]

def detrend(arr, level=7):
    return arr - decompose(arr, level)[-1]

def wavelet_enh_std(f, level=4, out = 'rec', absp = False):
    fw = dec_atrous2d(f, level)
    if absp:
        supp = map(lambda x: abs(x) > x.std(), fw)
    else:
        supp = map(lambda x: x > x.std(), fw)
    if out == 'rec':
        return rec_with_support(fw, supp)
    elif out == 'supp':
        return represent_support(supp)

def rec_with_support(coefs, supp):
    return rec_atrous([c*s for c,s in zip(coefs, supp)])


def qmf(filt = _phi_):
    L = len(filt)
    return [(-1)**(l+1)*filt[L-l-1] for l in range(len(filt))]
        
def wavelet_denoise(f, k=[3.5,3.0,2.5,2.0], level = 4, noise_std = None,
		    modulus=False,
		    soft=False):
    if np.iterable(k):
        level = len(k)
    coefs = decompose(f, level)
    if noise_std is None:
        if False and f.ndim < 3:
            noise_std = estimate_sigma(f, coefs) / 0.974 # magic value
        else:
            noise_std = estimate_sigma_mad(coefs[0], True)
    supp = get_support(coefs, np.array(k, _dtype_)*noise_std,
		       modulus=modulus,soft=soft)
    if soft:
	return np.sum(supp, axis=0)
    else:
	return rec_with_support(coefs, supp)

def DFoF(v, level=9):
    """DF/F0 for F0 taken as approximation at given level"""
    approx = smooth(v, level)
    zi = np.where(np.abs(approx) < 1e-6)
    approx[zi] = 1.0
    out = v/approx - 1.0
    out[zi] = 0
    return out

def DFoSD(v, level=9, smooth = 0):
    coefs = decompose(v, level)
    approx = coefs[-1]
    if smooth:
	vd = np.sum(coefs[smooth:-1], axis=0)
    else:
	vd = v-approx
    sd = estimate_sigma_mad(coefs[0], True)
    if sd == 0:
	return np.zeros(vd.shape)
    return vd/sd
    

