### Empirical mode decomposition routines

import numpy as np
from .. import core
from ..core import extrema

def envelopes(vec, x=None):
    """
    Given vector `vec` and optional `x`, return lower and upper envelopes defined
    by local extrema in vector vec.  Variable `x` defines the abscissa for
    vector `vec`
    """
    from scipy.interpolate import splrep,splev
    xfit,yfit,_,maxlocs,minlocs = extrema.locextr(vec,x,sort_values=False)
    if (len(maxlocs) < 1) or (len(minlocs) < 1):
	return []
    if x is None: x = np.arange(len(vec))
    minlocs = np.concatenate([[0],minlocs,[-1]])
    maxlocs = np.concatenate([[0],maxlocs,[-1]])
    #print len(minlocs), len(maxlocs)
    if min(*map(len,(minlocs,maxlocs))) < 4:
        return []
    lower = splev(x,splrep(xfit[minlocs],yfit[minlocs]))
    upper = splev(x,splrep(xfit[maxlocs],yfit[maxlocs]))
    return lower, upper

def mean_env(vec, x=None):
    '''
    Given vector `vec` and optional `x`, return a mean envelope
    '''
    envs = envelopes(vec,x)
    if len(envs):
	return np.mean(envelopes(vec,x), axis=0)
    else:
	return None

def imf_candidate(vec,x=None):
    '''
    Given vector `vec` and optional `x`, return a candidate for an IMF
    '''
    h = mean_env(vec,x)
    if h is None:
	return None
    else:
	return vec-mean_env(vec,x)

def find_mode(vec, x=None,SDk = 0.2,max_iter=1e5):
    """Finds first empirical mode of the vector `vec`
    returns mode if it can, returns ``None`` if no local extrema can be found

    Parameters:
      - `vec`: an input 1D vector
      - `x`: an optional vector, `vec=f(x)`
      - `SDk`: tolerance
      - `max_iter`: maximum number of iterations

    Returns:
      - `h`: a mode estimate

    """
    h_prev = imf_candidate(vec,x)
    if h_prev is None:
	return None
    for k in xrange(long(max_iter)):
	h1 = imf_candidate(h_prev, x)
	if h1 is None:
	    return h_prev
	sd = np.sum((h1-h_prev)**2)/np.sum(h_prev**2)
	xf,yf,der,mx,mn = extrema.locextr(h1)
	zc = np.where(np.diff(np.sign(h1)) > 0 )[0]
	#print len(zc), len(mx), len(mn)
	if (abs(len(zc) - len(mx)) < 2) and \
	   (abs(len(mx) - len(mn)) < 2) and \
	   (abs(len(zc) - len(mn)) < 2) and \
	   sd < SDk:
	    return h1
	h_prev = h1
    print "No convergence after %d iterations" % max_iter
    return h1

def find_all_modes(vec,x=None, max_modes = 100):
    """
    Iteratively run ``find_mode`` to obtain all empirical modes in vector `vec`

    Parameters:
      - `vec`: an input 1D vector
      - `x`: an optional vector, `vec=f(x)`
      - `max_modes`: maximum number of modes to look for

    Returns:
      a list of nodes and a remainder: `modes`, `rem`
    """
    import itertools as itt
    modes = []
    rem = np.copy(vec)
    for k in xrange(max_modes):
	h = find_mode(rem, x)
	if h is None:
	    return modes, rem
	else:
	    modes.append(h)
	    rem = rem - h
    return modes, rem
