### Monte-Carlo routines for significance level estimation

import numpy as np
import sys
 
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

def _mc_levels1d(transform_fn, size=1e5, level=12, N = 1e3):
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
    signals = (np.random.randn(size) for i in np.arange(N))
    out  = np.zeros((N,level))
    for n,im in enumerate(signals):
	x = np.mean(out[:n], axis=0)
	s0 = ','.join(['%1.4e'%a for a in x])
	s = '\r signal {:06d} out of {:06d}, current: {}'.format(n+1,long(N), s0)
	sys.stderr.write(s)
	#out[n] = map(np.std, decompose1d_direct(im, level)[:-1])
        out[n] = map(np.std, transform_fn(im, level)[:-1])
    return np.mean(out, axis=0)
