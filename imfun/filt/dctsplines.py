

import numpy as np

from numpy import cos, pi, arange, ones, zeros
from numpy.linalg import norm

from scipy.fftpack import dct,idct
from scipy import sparse

from ..core import fnutils

from numba import jit

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
    return np.where(x!=0, x*np.maximum(ax-gamma,0)/(1e-9+ax), 0)



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

    for nit_ in range(niter):
        z = idct(gamma*dct(weights*(v - zprev) + zprev, norm='ortho'),norm='ortho')
        if norm(z-zprev)/norm(zprev) < eps:
            if verbose:
                print('weights: reached convergence at %d iterations'%nit_)
            break
        zprev = z
    return z


def dct2d(m,norm='ortho'):
    return dct(dct(m, norm=norm, axis=0),
               norm=norm, axis=1)


def dctnd(m,norm='ortho'):
    fa = lambda a: lambda m_: dct(m_,norm=norm,axis=a)
    f = fnutils.flcompose(*list(map(fa, range(np.ndim(m)))))
    return f(m)

def idctnd(m,norm='ortho'):
    fa = lambda a: lambda m_: idct(m_,norm=norm,axis=a)
    f = fnutils.flcompose(*list(map(fa, range(np.ndim(m)))))
    return f(m)

def idct2d(m,norm='ortho'):
    return idct(idct(m, norm=norm, axis=0),
               norm=norm, axis=1)

def Lambda_nd(sh):
    ndims = len(sh)
    shapes = np.diag(np.array(sh)-1)+1
    Lams = [-2 + 2*cos((arange(n)*pi)/n) for n in sh]
    return sum(l.reshape(s) for l,s in zip(Lams, shapes)) # note this is python sum, not np.sum
    #indices = meshgrid(*(range(n) for n in sh))
    #return indices

def Gamma(sh,s):
    Lam = Lambda_nd(sh)
    o = np.ones(sh)
    return o/(o + s*Lam*Lam)



def l2spline(m, s, weights=None, eps=1e-3, niter=1000, s_is_scale=True, verbose=False,
             scale_converter=l2sp_gauss_scale_to_smooth):
    sh = m.shape
    if s_is_scale:
        s = scale_converter(s)
    g = Gamma(sh,s)
    zprev = idctnd(g*dctnd(m))
    if weights is None or np.allclose(weights, ones(sh)):
        return zprev

    for nit_ in range(niter):
        z = idctnd(g*dctnd(weights*(m - zprev) + zprev))
        if norm(z-zprev)/norm(zprev) < eps:
            if verbose:
                print('weights: reached convergence at %d iterations'%nit_)
            break
        zprev = z
    return z

from scipy import ndimage
def rolling_sd_scipy_nd(arr,hw=None,correct_factor=1.,smooth_output=True):
    if hw is None: hw = int(np.max(arr.shape)/10)
    padded = np.pad(arr,hw,mode='reflect')
    rolling_median = lambda x: ndimage.median_filter(x, 2*hw)
    crop = (slice(hw,-hw),)*np.ndim(arr)
    out = 1.4826*rolling_median(np.abs(padded-rolling_median(padded)))[crop]

    out = out/correct_factor
    if smooth_output:
        out = l2spline(out, s=2**np.ndim(arr)*hw)
    return out


def rolling_sd_scipy_nd(arr,hw=None,correct_factor=1.,smooth_output=True):
    if hw is None: hw = int(np.ceil(np.max(arr.shape)/10))
    padded = np.pad(arr,hw,mode='reflect')
    rolling_median = lambda x: ndimage.median_filter(x, 2*hw)
    crop = (slice(hw,-hw),)*np.ndim(arr)
    out = 1.4826*rolling_median(np.abs(padded-rolling_median(padded)))[crop]

    out = out/correct_factor
    if smooth_output:
        out = l2spline(out, s=hw)
    return out

def sigma_mad(v):
    return 1.4826*np.median(np.abs(v-np.median(v)))

def get_thresholded_dct(img, nharmonics=100, th=3):
    dct_coefs = dctnd(img)
    crop_in = (slice(None, nharmonics),) * np.ndim(img)
    cutoff_mask = np.zeros(img.shape, bool)
    cutoff_mask[crop_in] = True
    dct_coefs *= cutoff_mask
    if nharmonics >=10:
        nsm = rolling_sd_scipy_nd(dct_coefs[crop_in])
        dct_coefs[crop_in] *= abs(dct_coefs[crop_in]) > th*nsm
    return dct_coefs


def l2spline_thresholded(m, s, s_is_scale=True,
                         nharmonics = 100,
                         th = 3,
                         scale_converter=l2sp_gauss_scale_to_smooth):
    sh = m.shape
    if s_is_scale:
        s = scale_converter(s)
    g = Gamma(sh,s)
    dct_coefs = dctnd(m)
    crop_in = (slice(None, nharmonics),) * np.ndim(m)
    cutoff_mask = np.zeros(m.shape, bool)
    smoothed_coefs = np.zeros_like(dct_coefs)
    smoothed_coefs[crop_in] = g[crop_in]*dct_coefs[crop_in]
    if nharmonics >=10:
        nsm = rolling_sd_scipy_nd(dct_coefs[crop_in])
        #nsm = np.std(dct_coefs[~cutoff_mask])
        #nsm = sigma_mad(dct_coefs[crop_in])
        #print('nsm:', np.max(nsm))
        smoothed_coefs[crop_in] = np.where(abs(dct_coefs[crop_in])>th*nsm,
                                           dct_coefs[crop_in],
                                           smoothed_coefs[crop_in])
    return idctnd(smoothed_coefs)



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
            err = norm(z-zprev)/(1e-8 + norm(zprev))
            if err < eps:
                if verbose:
                    print('converged after',_i,'iterations')
                break
        zprev = np.copy(z)
    return z#, acc

def l1spline(m, s=25.0, lam=None, weights=None, eps=1e-3, Ni=1,  niter=1000,
             weight_niter = 100,
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
    noweights = False
    if weights is None or np.allclose(weights,ones(sh)):
        noweights = True
        weights = np.ones(sh)
    for _i in range(niter):
        x_ = d+m-b
        zprev_w = idctnd(g*dctnd(x_))
        if noweights:
            z = zprev_w
        else:
            for wi_ in range(weight_niter):
                z = idctnd(g*dctnd(weights*(x_-zprev_w)+zprev))
                if norm(z-zprev_w)/norm(zprev_w)<eps:
                    break
                zprev_w = z
        d = shrink((b+z-m),1.0/lam )
        b = b + weights*(z-m-d)
        #acc.append(map(copy, [z, d, b]))
        if _i >0:
            err = norm(z-zprev)/(1e-8+norm(zprev))
            if err < eps:
                if verbose:
                    print('converged after',_i,'iterations')
                break
        zprev = np.copy(z)
    return z#, acc


def l1spline_thresholded(m, s=25.0, lam=None,  eps=1e-3, Ni=1,
                         nharmonics = 100,
                         th = 3,
                         niter= 1000,
                         verbose=False,
                         scale_converter=l1sp_gauss_scale_to_smooth,
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
    noweights = True
    crop_in = (slice(None, nharmonics),) * np.ndim(m)

    for _i in range(niter):
        x_ = d+m-b
        coefs = dctnd(x_)
        smoothed_coefs = np.zeros_like(coefs)
        smoothed_coefs[crop_in] = g[crop_in]*coefs[crop_in]
        if nharmonics >=10:
            nsm = rolling_sd_scipy_nd(coefs[crop_in])
            smoothed_coefs[crop_in] = np.where(abs(coefs[crop_in])>th*nsm,
                                               coefs[crop_in],
                                               smoothed_coefs[crop_in])
        zprev_w = idctnd(smoothed_coefs)
        z = zprev_w
        d = shrink((b+z-m),1.0/lam )
        b = b + (z-m-d)
        if _i >0:
            err = norm(z-zprev)/(1e-8+norm(zprev))
            if err < eps:
                if verbose:
                    print('converged after',_i,'iterations')
                break
        zprev = np.copy(z)
    return z#, acc


def sp_decompose(s,level=24, base=2.,smoother=l1spline,s_is_scale=True):
    out = []
    if np.iterable(level):
        levels = level
    else:
        levels = (base**l for l in range(0,level))
    approx_prev = s
    for l in levels:
        approx = smoother(approx_prev, l, s_is_scale=s_is_scale)
        out.append(approx_prev-approx)
        approx_prev = approx
    return np.array(out + [approx_prev])
