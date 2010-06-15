import numpy as np
import time, sys
#from imfun.aux_utils import ifnot
from swan import pycwt

def ifnot(a, b):
    "if a is not None, return a, else return b"
    if a == None: return b
    else: return a

def isseq(obj):
    return hasattr(obj, '__iter__')

def cwtmap(fseq,
           tranges,
           frange,
           nfreqs = 32,
           wavelet = pycwt.Morlet(),
           func = np.mean,
           normL = None,
           kern=None,
           verbose = True,
           **kwargs):
    """
    Wavelet-based 'functional' map of the frame sequence
    
    Arguments
    ----------
    
    *extent* is the window of the form
    (start-time, stop-time, low-frequency, high-frequency)
    
    *nfreqs* -- how many different frequencies in the
    given range (default 16)
    
    *wavelet* -- wavelet object (default pycwt.Morlet())
    
    *func* -- function to apply to the wavelet spectrogram within the window
    of interest. Default, np.mean
    
    *normL* -- length of normalizing part (baseline) of the time series
    
    *kern* -- if 0, no alias, then each frame is filtered, if an array,
    use this as a kernel to convolve each frame with; see aliased_pix_iter
    for default kernel
    """
    tick = time.clock()
    L = fseq.length()
    shape = fseq.shape(kwargs.has_key('sliceobj') and
                       kwargs['sliceobj'] or None)
    total = shape[0]*shape[1]
    k = 0

    pix_iter = None
    normL = ifnot(normL, L)

    if not isseq(tranges[0]):
        tranges = (tranges,)
    
    if type(kern) == np.ndarray or kern is None:
        pix_iter = fseq.conv_pix_iter(kern,**kwargs)
    elif kern <= 0:
        pix_iter = pix_iter(**kwargs)

    freqs = np.linspace(*frange, num=nfreqs)

    tstarts,tstops = [],[]

    for tr in tranges:
        tstarts.append(int(tr[0]/fseq.dt)) # This is disgusting that I write in such way
        tstops.append(int(tr[1]/fseq.dt))

    out = np.ones((len(tranges),)+shape, np.float64)

    for s,i,j in pix_iter:
        s = s-np.mean(s[:normL])
        cwt = pycwt.cwt_f(s, freqs, 1./fseq.dt, wavelet, 'zpd')
        eds = pycwt.eds(cwt, wavelet.f0)/np.std(s[:normL])**2
        for tk, tr in enumerate(tranges):
            out[tk,i,j] = func(eds[:,tstarts[tk]:tstops[tk]])
        k+= 1
        if verbose:
            sys.stderr.write("\rpixel %05d of %05d"%(k,total))
    if verbose:
        sys.stderr.write("\n Finished in %3.2f s\n"%(time.clock()-tick))
    return out

def fftmap(fseq, frange, kern = None, func=np.mean,
           normL = None,
           verbose = True,
           **kwargs):
        """
        Fourier-based functional mapping
        frange : a range of frequencies in Hz, e.g. (1.0, 1.5)
        kern  : a kernel to convolve each frame with
        func  : range reducing function. np.mean by default, may be np.sum as well
        """
        tick = time.clock()

        L = fseq.length()

        shape = fseq.shape(kwargs.has_key('sliceobj') and
                           kwargs['sliceobj'] or None)

        total = shape[0]*shape[1]
        

        out = np.ones(shape, np.float64)

        k = 0
        freqs = np.fft.fftfreq(L, fseq.dt)

        pix_iter = None
        if type(kern) == np.ndarray or kern is None:
            pix_iter = fseq.conv_pix_iter(kern,**kwargs)
        elif kern <= 0:
            pix_iter = pix_iter(**kwargs)
        normL = ifnot(normL, L)
        
        fstart,fstop = frange
        fmask = (freqs >= fstart)*(freqs < fstop)
        for s,i,j in pix_iter:
            s = s-np.mean(s[:normL])
            s_hat = np.fft.fft(s)
            x = (abs(s_hat)/np.std(s[:normL]))**2
            out[i,j] = func(x[fmask])
            k+=1
            if verbose:
                sys.stderr.write("\rpixel %05d of %05d"%(k,total))
        if verbose:
            sys.stderr.write("\n Finished in %3.2f s\n"%(time.clock()-tick))
        return out
