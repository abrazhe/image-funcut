import numpy as np
from scipy.interpolate import splrep,splev

def in_range(v, low, high):
    if low > high:
        low,high = high,low
    return (v > low) * (v <= high)
    

def locextr(x, mode = 1, **kwargs):
   "Finds local maxima when mode = 1, local minima when mode = -1"
   tck = splrep(range(len(x)),x, **kwargs)
   res = 0.05
   xfit = np.arange(0,len(x), res)
   dersign = mode*np.sign(splev(xfit, tck, der=1))
   return xfit[dersign[:-1] - dersign[1:] > 1.5]

def all_vert_maxima(arr):
    return map(locextr, arr)

def limit_bounds(vec, lbound, ubound):
   out = np.copy(vec)
   if lbound:
      out = out[out >= lbound]
   if ubound:
        out = out[out <= ubound]
   return out
         

def trace(seq, seed, memlen = 2, memweight = 0.75,
          lbound = None, ubound = None):
   out = []
   seeds = seed*np.ones(memlen)
   for k,el in enumerate(seq):
      el = limit_bounds(el, lbound, ubound)
      if len(el) > 0:
         j = argmin(abs(el - np.mean(seeds)))
         seeds[:-1] = seeds[1:]
         seeds[-1] = el[j]
         out.append((k,el[j]))
   return array(out)

def ind_to_val(ind, vrange, nsteps, dv=None):
    if dv is None: dv = (vrange[1] - vrange[0])/nsteps
    return vrange[0] + ind*dv

def val_to_ind(val, vrange, nsteps, dv=None):
    if dv is None: dv = (vrange[1] - vrange[0])/nsteps
    return (val-vrange[0])/dv


def trace_hridge(arr, seed, extent, rhsd = 0.1):
    """
    trace horizontal ridge in array and return time, frequency and energy
    vectors
    """
    trange, frange = extent[:2], extent[2:]
    nfreqs, ntimes = arr.shape
    df = (frange[1] - frange[0])/nfreqs
    rhsd_i = rhsd/df

    a = map(locextr, arr.T)
    seed_i = val_to_ind(seed, frange, nfreqs)
    rhythm = trace(a, seed_i)

    tck = splrep(rhythm[:,0], rhythm[:,1])
    rhfreqs = np.asarray(map(int, np.round(splev(np.arange(ntimes), tck))))
    #fi1,fi2 = rhfreqs - rhsd_i, rhfreqs + rhsd_i

    fvec = ind_to_val(rhythm[:,1], frange, nfreqs)
    tvec = ind_to_val(rhythm[:,0], trange, ntimes)
    evec = np.asarray([np.sum(arr[rhfreqs[i]-rhsd:rhfreqs[i]+rhsd,i])
                       for i in xrange(ntimes)])
    return tvec, fvec, evec
                  
