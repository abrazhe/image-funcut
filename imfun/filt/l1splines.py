from __future__ import division

import numpy as np
from scipy.fftpack import dct,idct
from scipy import sparse



def shrink(x,gamma):
    """(x/|x|)*max(|x|-gamma)"""
    ax = np.abs(x)
    return np.where(x!=0, x*clip(ax-gamma,0,np.nan)/ax, 0)

def l1sp_scale_to_smooth(scale):
    return (scale**4)/1000.

def l1spline(y, s=25.0, lam=None, weights=None, eps=1e-3, Ni=1,  niter=1000,
             verbose=False,
             s_is_scale=True):
    n = len(y)
    if s_is_scale:
        s = l1sp_scale_to_smooth(s)
    if weights is None:
        weights = np.ones(n)
    if lam is None:
        lam = min(s,1)
    gamma = 1./(1. + s*(-2. + 2.*cos((np.arange(n)*pi/n)))**2)
    d = np.zeros(n)
    b = np.zeros(n)
    zprev = np.zeros(n)
    #acc = []
    for _i in range(niter):
        z = idct(gamma*dct((d+y-b),norm='ortho'),norm='ortho')
        d = shrink((b+z-y),1.0/lam )
        b = b + (z-y-d)
        #acc.append(map(copy, [z, d, b]))
        if _i >0:
            err = norm(z-zprev)/norm(zprev)
            if err < eps:
                if verbose:
                    print 'converged after',_i,'iterations'
                break
        zprev = copy(z)
    return z#, acc


def l1sp_pyramid(s,level=24, base=2.):
    out = []
    if np.iterable(level):
        levels = level
    else:
        levels = (base**l for l in xrange(1,level+1))
    approx_prev = s
    for l in levels:
        approx = l1spline(approx_prev, l)
        out.append(approx_prev-approx)
        approx_prev = approx
    return out + [approx_prev]
