### Functions for Multiscale median transform and variants

from scipy.ndimage import median_filter
import numpy as np

## TODO: 
## - [ ] implement pyramidal median transform

def decompose(arr, level,s=2):
    approx = median_filter(arr, 2*s+1)
    w = arr - approx # median coefficients
    if level == 1:
        return [w, approx]
    else:
        return [w] + decompose(approx, level-1, 2*s)
    

def MAD(x):
    m = np.median(x)
    return np.median(np.abs(x-m))

#    return np.median(np.abs(x - np.median(x)))

import atrous
def decompose_mwt(arr,level,s=2,tau=5, upto=3):
    """Median-Wavelet transform"""
    out = []
    cj = arr
    if level < upto:
        upto=level
    for j in range(upto):
        approxm = median_filter(cj, 2*s+1)
        wm = approxm - cj 
        th = tau*MAD(wm)/0.6745
        supp = np.abs(wm) > th
        if np.any(supp):
            wm[supp] = 0
        approx_dash = wm + approxm
        cjp1 = atrous.smooth(approx_dash, j+1)
        out.append(cj-cjp1)
        cj = cjp1
        s *= 2
    if level <= upto:
        out.append(cj)
        return out
    else:
        wcoefs = atrous.decompose(cj, level)
        im = np.sum(wcoefs[:upto+1], axis=0)
        return np.concatenate([out, [im], wcoefs[upto+1:]])
    #return out
    #wcoefs = atrous.decompose(cj, level)
    #for j in range(upto):
    #    wcoefs[j] = out[j]
    #return wcoefs
        
    
