import numpy as np
from swan import pycwt

def wfnmap(extent, nfreqs = 32,
           wavelet = pycwt.Morlet(),
           func = np.mean,
           normL = None,
           kern=None,
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
    
    shape = self.fseq.shape(kwargs.has_key('sliceobj') and
                                kwargs['sliceobj'] or None)
    out = ones(shape, np.float64)
    total = shape[0]*shape[1]
    k = 0
    freqs = linspace(*extent[2:], num=nfreqs)
    pix_iter = None
    normL = ifnot(normL, self.length())
    
    if type(kern) == np.ndarray or kern is None:
        pix_iter = self.fseq.conv_pix_iter(kern,**kwargs)
    elif kern <= 0:
        pix_iter = self.pix_iter(**kwargs)
        
    start,stop = [int(a/self.dt) for a in extent[:2]]
    for s,i,j in pix_iter:
        s = s-mean(s[:normL])
        cwt = pycwt.cwt_f(s, freqs, 1./self.dt, wavelet, 'zpd')
        eds = pycwt.eds(cwt, wavelet.f0)/std(s[:normL])**2
        x=func(eds[:,start:stop])
        #print "\n", start,stop,eds.shape, "\n"
        out[i,j] = x
        k+= 1
        if self.verbose:
            sys.stderr.write("\rpixel %05d of %05d"%(k,total))
    if self.verbose:
        sys.stderr.write("\n Finished in %3.2f s\n"%(time.clock()-tick))
    return out
