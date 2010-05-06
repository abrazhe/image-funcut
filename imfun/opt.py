# Optimization utilities

import numpy as np

def gaussian(x, (a, b, c,f)):
    return a * np.exp(-(x-b)**2/(2*c**2))+f

def fwhm(c):
    return np.sqrt(2*np.log(2))*c

def score_gauss(p, x, v):
    return sum((v-gaussian(x,p))**2)


from scipy import optimize

def gauss_fit(p0,x,y):
    return optimize.fmin(score_gauss, p0, (x, y))


def iter_gauss(p0, vectors, vslice=slice(None)):
    out = []
    for v in vectors:
        x = np.arange(len(v))[vslice]
        p0 = gauss_fit(p0,x,v[vslice])
        out.append(p0)
    return out
