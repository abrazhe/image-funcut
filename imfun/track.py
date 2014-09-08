import lib
import numpy as np
import scipy.interpolate as ip


import atrous

def limit_bounds(vec, lbound, ubound):
   out = np.copy(vec)
   if lbound:
      out = out[out >= lbound]
   if ubound:
        out = out[out <= ubound]
   return out

def gaussian(mu, sigma):
    return lambda _x: np.exp(-(_x-mu)**2/(2*sigma**2))


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

## def spl1st_derivative(ck):
##     L = len(ck)
##     return [0.5*(ck[mirrorpd(k+1,L)] - ck[mirrorpd(k-1,L)])  for k in range(L)]

## def spl2nd_derivative(ck):
##     L = len(ck)
##     return np.array([ck[mirrorpd(k+1,L)] + ck[mirrorpd(k-1,L)] - 2*ck[k] for k in range(L)])

def mirrorpd(k, L):
    if 0 <= k < L : return k
    else: return -(k+1)%L


def locextr(v, x=None, mode = 'max', refine=10, output='xfit'):
   "Finds local maxima when mode = 1, local minima when mode = -1"

   if type(x) is str:
       mode = x

   if x is None or type(x) is str:
       x = np.arange(len(v))
       
   sp0 = ip.UnivariateSpline(x,atrous.smooth(v),s=0)
   if mode in ['max', 'min']:
       sp = sp0.derivative(1)
   elif mode in ['gup', 'gdown', 'gany']:
       sp = sp0.derivative(2)
   res = 0.05
   if refine > 1:
       xfit = np.linspace(0,x[-1], len(x)*refine)
   else:
       xfit = x
   di = sp(xfit)
   if mode in ['max', 'gup']:
       dersign = np.sign(di)
   elif mode in ['min', 'gdown']:
       dersign = -np.sign(di)
   locations = dersign[:-1] - dersign[1:] > 1.5
   
   if output is 'all':
       out =  xfit[locations], sp0(xfit)[locations]
   elif output is 'yfit':
       out = di[locations]
   elif output is 'xfit':
       out = xfit[locations]
   else:
       print """unknown output code, should be one of  'xfit', 'yfit', 'all',
       returning 'x' locations"""
       out = xfit[locations]
   return out

def guess_seeds(seq, Nfirst=10):
    """
    automatically guess starting seeds for vessel walls
    relies on the fact that vessel is the largest bright stripe
    """
    Nfirst = min(len(seq), Nfirst)
    y = np.mean(seq[:Nfirst],axis=0)
    (xfit,yfit), (mx,mn), (gups,gdowns) = lib.extrema2(y, sort_values=True)
    # highest gradient up to the left of the highest max
    gu1 = (g for g in gups if g < mx[0]).next()
    # highest gradient up to the right of the highest max
    gd1 = (g for g in gdowns if g > mx[0]).next() 
    return (xfit[gu1], xfit[gd1])


def follow_extrema(arr, start, mode='gany', memlen=5):
    """
    modes: {'gup' | 'gdown' | 'min' | 'max' | 'gany'}
    ##(gany -- any gradient extremum, i.e. d2v/d2x =0)
    """
    def _ext(v):
        ##(xf,yf),(mx,mn), (gups, gdowns) = lib.extrema2(v)
        if mode in ['gup', 'gdown', 'max', 'min']:
            return locextr(v, mode=mode)
        elif mode =='gany':
            return np.concatenate([locextr(v,mode='gup'),locextr(v,mode='gdown')])
    extrema = map(_ext, arr)
    v = track1(extrema, start, memlen=memlen)
    return v


def track_pkalman(xv,seq, seed_mu,seed_var=4.,gain=0.25):
    history = seed_mu #seed_mu*np.ones(memlen)
    delta=0
    out = []
    for el in seq:
        mu = history
        # broadness of window depends on previous scatter in positions and
        # inversly on SNR in the current frame (within a window around
        # predicted position
        snr = np.max(el)/np.std(el)
        var = seed_var + delta**2 + (1./snr)**2
        p = gaussian(mu,var**0.5)(xv)*el
        z = xv[np.argmax(p)]
        g = (1+snr)/(1+snr/gain)
        x = g*mu + (1-g)*z
        history = x
        delta = mu-x
        out.append(x)
    return np.array(out)
    
def v2grads(v):
    L = len(v)
    sp2 = ip.UnivariateSpline(np.arange(L),atrous.smooth(v), s=0)
    xfit = np.arange(0,L,0.1)
    return np.abs(sp2.derivative(1)(xfit))

def track_walls(linescan,output = 'kalman',Nfirst=None,gain=0.25):
    '''
    output can be one of ('kalman',  'extr', 'mean', 'all')
    '''
    if Nfirst is None: Nfirst = len(linescan)/2
    seeds = guess_seeds(linescan,Nfirst)
    xfit = np.arange(0,linescan.shape[1],0.1)
    grads = np.array(map(v2grads, linescan))
    if output in ['kalman', 'mean', 'all']:
        tk1,tk2 = [track_pkalman(xfit, grads,seed,gain=gain) for seed in seeds]
    if output in ['extr', 'mean', 'all']:
        te1,te2 = [follow_extrema(linescan, seed)[:,1] for seed in seeds] 
    if output == 'mean':
        return 0.5*(tk1+te1), 0.5*(tk2+te2)
    elif output == 'kalman':
        return tk1,tk2
    elif output == 'extr':
        return te1,te2
    elif output == 'all':
        return tk1,tk2, te1,te2
