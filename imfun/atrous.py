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


## this is used for noise estimation and support calculation
sigmaej = [[0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],   # 0D 
           [0.700, 0.323, 0.210, 0.141, 0.099, 0.071, 0.054],   # 1D
           [0.889, 0.200, 0.086, 0.041, 0.020, 0.010, 0.005],   # 2D
           [0.956, 0.120, 0.035, 0.012, 0.004, 0.001, 0.0005]] # 3D



## Default spline wavelet scaling function
_phi_ = np.array([1./16, 1./4, 3./8, 1./4, 1./16])

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
    "1d stationary wavelet transform"
    apprx = convolve1d(sig, phi, mode='mirror')
    w = (sig - apprx) # wavelet coefs
    if level <= 0: return sig
    elif level == 1 or L < len(zupsample(phi)): return [w, apprx]
    else: return [w] + decompose1d(apprx, level-1, zupsample(phi))

def decompose2d(arr2d, level, kern=None, boundary='symm'):
    """
    2d stationary wavelet transform with B3-spline scaling function

    This is a convolution version, where kernel is zero-upsampled
    explicitly. Not fast.

    Arguments:
    ---------
    arr2d : 2D array
    level : level of decomposition

    Keyword arguments:
    -------------------
    kern  : low-pass filter kernel (B3-spline by default)
    boundary : boundary conditions (passed to scipy.signal.convolve2d, 'symm'
               by default)
    Outputs:
    ---------
    list of wavelet details + last approximation
    
    """
    _b3spline1d = np.array(_phi_)
    __x = _b3spline1d.reshape(1,-1)
    _b3spl2d = np.dot(__x.T,__x)
    if kern is None: kern = _b3spl2d
    if level <= 0: return arr2d
    shapecheck = map(lambda a,b:a>b, arr2d.shape, kern.shape)
    assert np.all(shapecheck)
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

def decompose3d(arr, level=1,
                  phi = _phi_,
                  boundary = 'mirror'):
    "Semi-separable \'a trous wavelet decomposition for 3D data"
    phi2d = f2d(phi)
    if level <= 0: return arr
    tapprox = np.zeros(arr.shape)
    for loc in locations(arr.shape[1:]):
        v = arr[:,loc[0], loc[1]]
        tapprox[:,loc[0], loc[1]] = convolve1d(v, phi, mode=boundary)
    approx = np.zeros(arr.shape)
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

def get_support(coefs, th, neg=False):
    out = []
    nd = len(coefs[0].shape)
    fn = neg and np.less or np.greater
    for j,w in enumerate(coefs[:-1]):
	if np.iterable(th): t = th[j]
	else: t = th
	sj= sigmaej[nd][j]
	if np.iterable(t):
	    wa = np.abs(w)
	    out.append((wa > t[0]*sj)*(wa<=t[1]*sj))
	else:
	    out.append(fn(np.abs(w), t*sj))
    out.append(np.ones(coefs[-1].shape)*(not neg))
    return out


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

        
def wavelet_denoise(f, k=[3.5,3.0,2.5,2.0], level = 4, noise_std = None):
    if np.iterable(k):
        level = len(k)
    coefs = decompose(f, level)
    if noise_std is None:
        if f.ndim < 3:
            noise_std = estimate_sigma(f, coefs) / 0.974 # magic value
        else:
            noise_std = estimate_sigma_mad(coefs[0])
    supp = get_support(coefs, np.array(k)*noise_std)
    return rec_with_support(coefs, supp)
