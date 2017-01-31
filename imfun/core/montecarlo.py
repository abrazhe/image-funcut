### Monte-Carlo routines for significance level estimation

import numpy as np
import sys

from .random_proc import ar1
from .fnutils import take
 
def mc_levels(transform_fn, size=(256,256),level=3, N = 1e3):
    """Return Monte-Carlo estimation of noise :math:`\\sigma`

    Parameters:
      - transform_fn: (`function') -- decomposition transformation to use
      - size: (`tuple`) -- size of random noisy images
      - level: (`int`) -- level of decomposition
      - N: (`num`) -- number of random images to process

    Returns:
      - 1 :math:`\\times` level vector of noise :math:`\\sigma` estimations
    """
    import sys, pprint
    images = (np.random.randn(*size) for i in np.arange(N))
    out  = np.zeros((N,level))
    for n,im in enumerate(images):
       if n > 1:
          x = np.mean(out[:n], axis=0)
          s = ('%1.4f, '*len(x))%tuple(x)
          sys.stderr.write('\r image %06d out of %d, current: %s'%(n+1,N, s))
          #print n+1, 'current: ', x
       out[n] = map(np.std, transform_fn(im, level)[:-1])
    return np.mean(out, axis=0)

def _mc_levels1d(transform_fn, size=1e5, level=12, N = 1e3, noise_model='white',p=99.5):
    """Return Monte-Carlo estimation of noise :math:`\\sigma`

    Parameters:
      - transform_fn: (`function') -- decomposition transformation to use
      - size: (`tuple`) -- size of random signals
      - level: (`int`) -- level of decomposition
      - N: (`num`) -- number of random images to process

    Returns:
      - 1 :math:`\\times` level vector of noise :math:`\\sigma` estimations
    """
    import sys
    import itertools as itt
    from matplotlib import mlab
    N = int(N)
    size=int(size)
    if noise_model == 'white':
        signals = (np.random.randn(size) for i in np.arange(N,dtype=np.int))
    elif noise_model == 'ar1':
        signals = (np.array(take(size, ar1())) for i in np.arange(N,dtype=np.int))
    elif noise_model == 'ma':
        signals = (mlab.movavg(np.random.randn(size+5),5)[:size] for i in np.arange(N,dtype=np.int))
        signals = itt.imap(lambda s: s/np.std(s), signals)
    out  = np.zeros((N,level))
    for n,im in enumerate(signals):
	x = np.mean(out[:n], axis=0)
	s0 = ','.join(['%1.4e'%a for a in x])
        pdone = 100*(n+1)/float(N)
	s = '\r [{:02.1f}%] signal {:06d} of {:06d}, current: {}'.format(pdone, n+1,long(N), s0)
	sys.stderr.write(s)
	#out[n] = map(np.std, decompose1d_direct(im, level)[:-1])
        #out[n] = map(np.std, transform_fn(im, level)[:-1])
        out[n] = map(lambda v_: np.percentile(v_, p), transform_fn(im, level)[:-1])
    return np.mean(out, axis=0)
