import lib
import numpy as np
from scipy.interpolate import splrep,splev
from scipy.signal import cspline1d


def limit_bounds(vec, lbound, ubound):
   out = np.copy(vec)
   if lbound:
      out = out[out >= lbound]
   if ubound:
        out = out[out <= ubound]
   return out


def track1(seq, seed, memlen = 5, 
           guide = None, 
           lbound = None, ubound = None):
    out = []
    seeds = seed*np.ones(memlen)
    use_guide = (guide is not None and len(guide) >= len(seq))
    for k,el in enumerate(seq):
        el = limit_bounds(el, lbound, ubound)
        if len(el) > 0:
            if use_guide: 
                target = (np.sum(seeds) + guide[k])/(len(seeds)+1)
            else:
                target = np.mean(seeds)
            j = np.argmin(abs(el - target))
            seeds[:-1] = seeds[1:]
            seeds[-1] = el[j]
            out.append((k,el[j]))
    return np.array(out)

def spl1st_derivative(ck):
    L = len(ck)
    return [0.5*(ck[mirrorpd(k+1,L)] - ck[mirrorpd(k-1,L)])  for k in range(L)]

def spl2nd_derivative(ck):
    L = len(ck)
    return np.array([ck[mirrorpd(k+1,L)] + ck[mirrorpd(k-1,L)] - 2*ck[k] for k in range(L)])

def mirrorpd(k, L):
    if 0 <= k < L : return k
    else: return -(k+1)%L


def locextr(x, mode = 'max', **kwargs):
   "Finds local maxima when mode = 1, local minima when mode = -1"
   ck = cspline1d(x, 2) #smooth spline coefs
   d1ck = spl1st_derivative(ck) # first derivative
   d2ck = spl2nd_derivative(ck) # second derivative
   if mode in ['max', 'min']:
       tck = splrep(range(len(x)), d1ck, **kwargs)
   elif mode in ['gup', 'gdown', 'gany']:
       tck = splrep(range(len(x)), d2ck, **kwargs)
   res = 0.05
   xfit = np.arange(0,len(x), res)
   di = splev(xfit, tck)
   if mode in ['max', 'gup']:
       dersign = np.sign(di)
   elif mode in ['min', 'gdown']:
       dersign = -np.sign(di)
   return xfit[dersign[:-1] - dersign[1:] > 1.5]


def follow_extrema(arr, start, mode='gany', memlen=5):
    """
    modes: {'gup' | 'gdown' | 'min' | 'max' | 'gany'}
    ##(gany -- any gradient extremum, i.e. d2v/d2x =0)
    """
    def _ext(v):
        ##(xf,yf),(mx,mn), (gups, gdowns) = lib.extrema2(v)
        if mode in ['gup', 'gdown', 'max', 'min']:
            return locextr(v, mode)
        elif mode =='gany':
            return np.concatenate([locextr(v,'gup'),locextr(v,'gdown')])
    extrema = map(_ext, arr)
    v = track1(extrema, start, memlen=memlen)
    return v
