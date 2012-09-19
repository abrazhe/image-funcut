### Functional mapping tools 


import numpy as np
import time, sys
from swan import pycwt

from imfun import lib, bwmorph
ifnot = lib.ifnot

neighbours = bwmorph.neighbours
locations = bwmorph.locations

def isseq(obj):
    '''Simple test if an object is a sequence (has "__iter__" attribute)'''
    return hasattr(obj, '__iter__')


def cwt_iter(fseq,
             frange,
             nfreqs = 128,
             wavelet = pycwt.Morlet(),
             normL = None,
             max_pixels = None,
	     cwtfn = pycwt.eds,
             verbose = False,
             **kwargs):
    """
    Iterate over cwt of the time series for each pixel

    Parameters:
      - `fseq` -- frame sequence instance
      - `frange`  -- frequency range as a pair or vector of frequencies
      - `nfreqs` -- number of frequencies/scales for decomposition 
      - `wavelet` -- wavelet object [pycwt.Morlet()]
      - `normL` -- length of normalizing part (baseline) of the time series 
      - `max_pixels` -- upper limit on number of pixels to iterate over 
      - `cwt_fn` -- function to process wavelet coefficients [pycwt.eds]
      - `verbose` -- be verbose 
      - `**kwargs` --  are passed to fseq.pix_iter

    Returns:
     - generator over (cwt-derived measure, i, j) tuples
      (where i,j are frame indices)
    """
    tick = time.clock()
    L = fseq.length()
    subframe = kwargs.has_key('sliceobj') and kwargs['sliceobj'] or None
    shape = fseq.shape(subframe)
    npix = shape[0]*shape[1]
    normL = ifnot(normL, L)
    pixel_iter = fseq.pix_iter(**kwargs)
    max_pixels = ifnot(max_pixels, npix)

    if len(frange) == 2:  # a low-high pair
        freqs = np.linspace(frange[0], frange[1], num=nfreqs)
    else:
        freqs= np.array(frange.copy())
    nfreqs = len(freqs)

    pixel_counter = 0
    npix = min(npix, max_pixels)
    cwtf = pycwt.cwt_f
    for s,i,j in pixel_iter:
        s = (s-np.mean(s[:normL]))/np.std(s[:normL])
        eds = pycwt.eds(cwtf(s, freqs, 1./fseq.dt, wavelet, 'zpd'))
        pixel_counter+= 1
        if verbose:
            sys.stderr.write("\rpixel %05d of %05d"%(pixel_counter,npix))
        yield eds, i, j
        if pixel_counter > max_pixels:
            break
    if verbose:
        sys.stderr.write("\n Finished in %3.2f s\n"%(time.clock()-tick))

def cwtmap(fseq,
           tranges,
           frange,
           func = np.mean,
           **kwargs):
    """
    Wavelet-based 'functional' map of the frame sequence
    
    Parameters:
     * `fseq` -- frame sequence
     * `tranges` -- list of time ranges
     * `frange` -- frequency range or vector of frequencies
     * `func` -- function to apply to the wavelet spectrogram within the window
              of interest. Default, np.mean
     * `**kwarg`s -- to be passed to cwt_iter

    Returns:
      - a 2D array as a result of application of the `func` to wavelet spectrograms
    """
    subframe = kwargs.has_key('sliceobj') and kwargs['sliceobj'] or None
    shape = fseq.shape(subframe)

    tstarts = map(lambda x: int(x[0]/fseq.dt), tranges)
    tstops = map(lambda x: int(x[1]/fseq.dt), tranges)

    out = np.zeros((len(tranges),)+shape, np.float64)
    for eds,i,j in cwt_iter(fseq,frange,**kwargs):
        for tk, tr in enumerate(tranges):
            out[tk,i,j] = func(eds[:,tstarts[tk]:tstops[tk]])
    return out

def cwtphase_map(fseq, ):
    pass

def loc_max_pos(v):
    x = range(0,len(v))
    p = np.polyfit(x,v,1)
    vx = v - np.polyval(p,x)
    vx[1:]-vx[:-1] > 0
    return [i for i in xrange(1,len(v)-1)
            if (vx[i] > vx[i-1]) and (vx[i] > vx[i+1])]

def cwt_freqmap(fseq,
                tranges,
                frange,
                nfreqs = 32,
                **kwargs):
    """
    Maps dominant frequency in the given frequency band, creates
    maps for a list of specified time ranges
    """
    from matplotlib import mlab
    if len(frange) > 2:
        freqs = frange
    else:
        freqs = np.linspace(frange[0], frange[-1],nfreqs)
    def _dominant_freq(arr):
        ma = np.mean(arr,1) 
        if np.max(ma) < 1e-7:
            print "mean wavelet power %e too low"%np.mean(ma)
            return np.nan
	i = np.argmax(mlab.detrend_linear(ma))
	return freqs[i]
    return cwtmap(fseq,tranges,freqs,func=_dominant_freq,**kwargs)


def avg_eds(fseq, *args, **kwargs):
    """
    Average wavelet energy density surface for a frame sequence
    input: fseq, `*args`, `**kwargs`
    `*args`, `**kwargs` are passed to cwt_iter
    """
    cwit = cwt_iter(fseq, *args, **kwargs)
    out,i,j = cwit.next()
    counter = 1.0
    for eds, i, j in cwit:
        out += eds
        counter += 1
    return out/counter

def _feature_map(fseq, rhythm, freqs, **kwargs):
    from scipy.interpolate import splrep,splev
    subframe = kwargs.has_key('sliceobj') and kwargs['sliceobj'] or None
    shape = fseq.shape(subframe)

    L = fseq.length()
    tinds = np.arange(L)
    tck = splrep(rhythm[:,0], rhythm[:,1])
    rhfreqs = map(int, np.round(splev(tinds, tck)))
    rhsd = 6

    out = np.ones((L,)+shape, np.float64)
    for eds,i,j in cwt_iter(fseq,freqs,**kwargs):
        for k in tinds:
            fi1, fi2  = rhfreqs[k] - rhsd, rhfreqs[k] + rhsd
            out[k,i,j] = np.sum(eds[fi1:fi2,k])
        #out[:,i,j] /= eds.mean()
    return out
    

def tanh_step(start,stop):
    "To be used for correlation"
    def _(t):
        v =  0.5*(1 + np.tanh(10*(t-start)) * np.tanh(-10*(t-stop)))
        return v - np.mean(v)
    return _

def MH_onoff(start,stop):
    "To be used for correlation"
    mh = pycwt.Mexican_hat()
    w = stop-start
    scale = 2*w*mh.fc
    def _(t):
        v = mh.psi((t - (start+0.5*w))/scale)
        return v
    return _

DFoSD = lib.DFoSD # Simple normalization function

def detrend(y, ord=2, take=None):
    x = np.arange(len(y))
    if take is None:
        take = x
    p = np.polyfit(x[take],y[take],ord)
    return y - np.polyval(p, x)


def meanactmap(fseq, (start,stop), normL=None):
    "Average activation map"
    L = fseq.length()
    normL = ifnot(normL, L)
    out = np.zeros(fseq.shape())
    tv = fseq.timevec()
    mrange = (tv > start)*(tv < stop)
    for s,j,k in fseq.pix_iter():
        sx = detrend(s,take=range(330))
        out[j,k] = np.mean(DFoSD(s,normL)[mrange])
    return out
    

def actcorrmap(fseq, (start, stop), normL=None,
               normfn = lib.DFoSD,
               sigfunc = tanh_step):
    "Activation correlation-based mapping"
    comp_sig = sigfunc(start, stop)(fseq.timevec())
    return xcorrmap(fseq, comp_sig, normL, normfn)

from scipy import stats

_corrfuncs = {'pearson':stats.pearsonr,
             'spearman':stats.spearmanr,
             'correlate':np.correlate} 

def xcorrmap(fseq, signal, normL=None, normfn = lib.DFoSD,
             corrfn = 'pearson',
             keyfn = lambda x:x[0],
	     normalize_data = False,
             normalize_signal=False):
    """
    Arguments:
      - fseq, a frame sequence instance
      - signal: a template signal (to correlate to)
      - normL : N points to use for normalization
      - normfn: a function to use for normalization [default: DFoSD]
      - corrfn: a correlation function. can be {'pearson', 'spearman', 'correlate'} 
        or user-provided function
      - keyfn : a function to extract corr. coefficient from corrfn returned value.
        default, [x->x]
      - normalize_signal:  flag, whether to normalize the template signal [default: False]

    Returned value:
     2D array, each value contains correlation coefficient of the provided frame
     sequence to the template signal. 
    """
    if type(corrfn) == str:
        corrfn = _corrfuncs[corrfn]
    out = np.zeros(fseq.shape())
    if normalize_signal:
        signal = normfn(signal, normL)
    for s,j,k in fseq.pix_iter():
	if normalize_data:
	    out[j,k] = keyfn(corrfn(normfn(s,normL), signal))
	else:
	    out[j,k] = keyfn(corrfn(s, signal))
    return out

def corrlag(timevec):
    """Factory: takes time vector. Returns a function,
    which takes two vectors, returns position and amplitude
    of the main peak in cross-correlation function
    """
    from imfun import atrous, lib
    tv1 = tv1 = np.concatenate((-timevec[::-1], timevec[1:]))
    def _(v1,v2):
	v1n = lib.DFoSD(v1)/len(v1)
	xcorr = np.correlate(v1n, lib.DFoSD(v2), mode='full')
	xcorr = atrous.smooth(xcorr, 2)
	maxima = lib.locextr(xcorr, x=tv1, output='max')
	k = np.argmax([x[1] for x in maxima])
	return maxima[k]
    return _


def xcorr_lag(fseq, **kwargs):
    from imfun import atrous
    tv = fseq.timevec()
    return xcorrmap(fseq, signal, corrfn=corrlag(tv),
		    **kwargs)

def local_corr_map(arr, normfn=lib.DFoSD,
                   corrfn=np.correlate,
                   keyfn=lambda x:x[0],
                   verbose=False):
    if type(corrfn) == str:
        corrfn = _corrfuncs[corrfn]
    sh = arr.shape[1:]
    out = np.zeros(sh)
    pixel_counter,npix = 0,np.prod(sh)
    def _v(l):
        v =  arr[:,l[0],l[1]]
	if normfn:
	    return normfn(v)
	else:
	    return v
    for loc in locations(sh):
        pixel_counter+= 1
        if verbose and not (pixel_counter % 100):
            sys.stderr.write("\rpixel %05d of %05d"%(pixel_counter,npix))
        local_corr = [keyfn(corrfn(_v(loc), _v(n))) for n in neighbours(loc,sh)]
        out[loc] = np.mean(local_corr)
    return out

    


def fftmap(fseq, frange, func=np.mean,
           normL=None,
           verbose=True,
           **kwargs):
        """
        Fourier-based functional mapping:
         - frange : a range of frequencies in Hz, e.g. (1.0, 1.5)
         - kern  : a kernel to convolve each frame with
         - func  : range reducing function. np.mean by default, may be np.sum as well
        """
        tick = time.clock()
        L = fseq.length()
        shape = fseq.shape(kwargs.has_key('sliceobj') and
                           kwargs['sliceobj'] or None)
        total = shape[0]*shape[1]
        out = np.ones(shape, np.float64)
        k = 0
        freqs = np.fft.fftfreq(L, fseq.dt)
        pix_iter = fseq.pix_iter(**kwargs)
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

            

    
