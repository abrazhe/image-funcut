### Functions for Multiscale median transform and variants

from scipy.ndimage import median_filter
import numpy as np

## TODO: 
## - [ ] implement pyramidal median transform

# when s=2 in decompose_mwt
sigmaej_mwts2 = [[], # 0D
                 [], # 1D]
                 [0.8908, 0.2111, 0.0808, 0.0343, 0.0200, 0.0107]] #2D

# when s=1 in decompose_mwt
sigmaej_mwts1 = [[], # 0D
                 [], # 1D]
                 [1.012,  0.229,  0.072,  0.044,  0.028,   0.015,]] #2D



## s=2 to match level-to-level resolution with starlet transform          
def decompose(arr, level, s=2):
    """Multiscale median transform decomposition"""
    approx = median_filter(arr, 2*s+1)
    w = arr - approx # median coefficients
    if level <= 0:
        return arr
    elif level == 1:
        return [w, approx]
    else:
        return [w] + decompose(approx, level-1, 2*s)
    

def MAD(x):
    m = np.median(x)
    return np.median(np.abs(x-m))

#    return np.median(np.abs(x - np.median(x)))


import atrous
def decompose_mwt(arr,level, s=2, tau=5, upto=2):
    """Median-Wavelet transform (up to level 'upto', then followed by starlet transform"""
    out = []
    cj = arr
    upto = min(level, upto)
    for j in range(upto):
        approxm = median_filter(cj, 2*s+1)
        wm = cj-approxm
        #th = tau*MAD(wm)/0.6745
        th = tau*MAD(wm[wm!=0])/0.6745
        wm[np.abs(wm) > th] = 0
        #wm *= np.abs(wm) <= th
        approx_dash = wm + approxm
        cjp1 = atrous.smooth(approx_dash, j+1)
        out.append(cj-cjp1)
        cj = cjp1
        s *= 2
    out.append(cj)
    if level > upto:
        wcoefs = atrous.decompose(cj, level)
        out[-1] = np.sum(wcoefs[:upto+1], axis=0)
        out = np.concatenate([out, wcoefs[upto+1:]])
    return out

def _asymmetric_smooth(v, level=9, niter=10, smooth=0, tol=1e-7):
    pass
    

# def threshold_w(coefs, th, neg=False, modulus=True,sigmaej=sigmaej_mwts2):
#     out = []
#     nd = len(coefs[0].shape)
#     fn = neg and np.less or np.greater
#     N = len(coefs)-1
#     for j,w in enumerate(coefs[:-1]):
#         if np.iterable(th): t = th[j]
# 	else: t = th
#         #if len(sigmaej[nd]):
#         sj = sigmaej[nd][j]
#         if modulus: wa = np.abs(w)
#         else: wa = w
#         #mask = fn(wa, (N-j)*th*MAD(w)/0.6745)
#         mask = fn(wa, t*sj)
#         out.append(mask)
#     out.append(np.ones(coefs[-1].shape)*(not neg))
#     return out
        
     
        
    
