from __future__ import division

import numpy as np

from numpy import cos, pi, arange, ones, zeros
from numpy.linalg import norm

from scipy.fftpack import dct,idct
from scipy import sparse

from ..core import fnutils

def l1sp_scale_to_smooth(scale,p_=(4,-3)):
    return scale**p_[0]*10**p_[1]

def l1sp_gauss_scale_to_smooth(scale,p_=(4, -0.25)):
    return scale**p_[0]*10**p_[1]

#  3.90501563 -0.24575408
def l2sp_scale_to_smooth(scale,p_=(3.9816815, -4.62098735)):
    return 10**np.polyval(p_, np.log10(scale))

# 4.0005062  -4.35526549
# [3.96, -4.28]
#[3.96, -0.22]

def l2sp_gauss_scale_to_smooth(scale,p_=(3.99006785, -0.25700078)):
    return 10**np.polyval(p_, np.log10(scale))

def shrink(x,gamma):
    """(x/|x|)*max(|x|-gamma)"""
    ax = np.abs(x)
    return np.where(x!=0, x*np.clip(ax-gamma,0,np.nan)/ax, 0)



from scipy.fftpack import dct,idct
from scipy import sparse



def l2spline1d(v, s=25., weights=None, eps=1e-3, niter=1000, s_is_scale=True,
             verbose=False,
             scale_converter=l2sp_gauss_scale_to_smooth):
    N = len(v)
    if s_is_scale:
        s = scale_converter(s)
    gamma = 1./(1. + s*(-2. + 2.*cos((np.arange(N)*pi/N)))**2)
    zprev = idct(gamma*dct(v,norm='ortho'),norm='ortho')
    if weights is None or np.allclose(weights, ones(N)):
        return zprev
    
    for nit_ in xrange(niter):
        z = idct(gamma*dct(weights*(v - zprev) + zprev, norm='ortho'),norm='ortho')
        if norm(z-zprev)/norm(zprev) < eps:
            if verbose:
                print 'weights: reached convergence at %d iterations'%nit_
            break
        zprev = z
    return z


def dct2d(m,norm='ortho'):
    return dct(dct(m, norm=norm, axis=0), 
               norm=norm, axis=1)  


def dctnd(m,norm='ortho'):
    fa = lambda a: lambda m_: dct(m_,norm=norm,axis=a)
    f = fnutils.flcompose(*map(fa, xrange(np.ndim(m))))
    return f(m)
    
def idctnd(m,norm='ortho'):
    fa = lambda a: lambda m_: idct(m_,norm=norm,axis=a)
    f = fnutils.flcompose(*map(fa, xrange(np.ndim(m))))
    return f(m)    

def idct2d(m,norm='ortho'):
    return idct(idct(m, norm=norm, axis=0), 
               norm=norm, axis=1)    

def Lambda_nd(sh):
    ndims = len(sh)
    shapes = np.diag(np.array(sh)-1)+1
    Lams = [-2 + 2*cos((arange(n)*pi)/n) for n in sh]
    return np.sum([l.reshape(s) for l,s in zip(Lams, shapes)],0)
    #indices = meshgrid(*(range(n) for n in sh))
    
    return indices

def Gamma(sh,s):
    Lam = Lambda_nd(sh)
    o = np.ones(sh)
    return o/(o + s*Lam*Lam)
    
    

def l2spline(m, s, weights=None, eps=1e-3, niter=1000, s_is_scale=True, verbose=True,
             scale_converter=l2sp_gauss_scale_to_smooth):
    sh = m.shape
    if s_is_scale:
        s = scale_converter(s)
    g = Gamma(sh,s)
    zprev = idctnd(g*dctnd(m))
    if weights is None or np.allclose(weights, ones(N)):
        return zprev
    
    for nit_ in xrange(niter):
        z = idct(g*dct(weights*(m - zprev) + zprev, norm='ortho'),norm='ortho')
        if norm(z-zprev)/norm(zprev) < eps:
            if verbose:
                print 'weights: reached convergence at %d iterations'%nit_
            break
        zprev = z
    return z
    


def l1spline1d(y, s, lam=None, weights=None, eps=1e-3, Ni=1,  niter=1000,
             verbose=False,scale_converter=l1sp_gauss_scale_to_smooth,
             s_is_scale=True):
    n = len(y)
    if s_is_scale:
        s = scale_converter(s)
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
        zprev = np.copy(z)
    return z#, acc

def l1spline(m, s, lam=None, weights=None, eps=1e-3, Ni=1,  niter=1000,
             verbose=False, scale_converter=l1sp_gauss_scale_to_smooth,
             s_is_scale=True):
    
    sh = m.shape
    if s_is_scale:
        s = scale_converter(s)
    if lam is None:
        lam = min(s,1)
    g = Gamma(sh,s)

    d = np.zeros(sh)
    b = np.zeros(sh)
    zprev = np.zeros(sh)
    #acc = []
    for _i in range(niter):
        z = idctnd(g*dctnd((d+m-b)))
        d = shrink((b+z-m),1.0/lam )
        b = b + (z-m-d)
        #acc.append(map(copy, [z, d, b]))
        if _i >0:
            err = norm(z-zprev)/norm(zprev)
            if err < eps:
                if verbose:
                    print 'converged after',_i,'iterations'
                break
        zprev = np.copy(z)
    return z#, acc


def sp_decompose(s,level=24, base=2.,smoother=l1spline,s_is_scale=True):
    out = []
    if np.iterable(level):
        levels = level
    else:
        levels = (base**l for l in xrange(0,level))
    approx_prev = s
    for l in levels:
        approx = smoother(approx_prev, l, s_is_scale=s_is_scale)
        out.append(approx_prev-approx)
        approx_prev = approx
    return np.array(out + [approx_prev])
