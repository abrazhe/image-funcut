import numpy as np
import time, sys
#from imfun.aux_utils import ifnot
from swan import pycwt

import itertools as itt

## def ifnot(a, b):
##     "if a is not None, return a, else return b"
##     if a == None: return b
##     else: return a

from imfun import lib, ui
ifnot = lib.ifnot

def isseq(obj):
    return hasattr(obj, '__iter__')

def cwt_iter(fseq,
             frange,
             nfreqs = 128,
             wavelet = pycwt.Morlet(),
             normL = None,
             max_pixels = None,
             verbose = True,
             **kwargs):
    """
    Iterate over cwt of the time series for each pixel
    *fseq* -- frame sequence
    *frange* -- frequency range or vector of frequencies
    *wavelet* -- wavelet object (default pycwt.Morlet())
    *normL* -- length of normalizing part (baseline) of the time series

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
    
    Arguments
    ----------
    *fseq* -- frame sequence
    *tranges* -- list of time ranges
    *frange* -- frequency range or vector of frequencies
    *func* -- function to apply to the wavelet spectrogram within the window
              of interest. Default, np.mean
    **kwargs -- to be passed to cwt_iter
    """
    subframe = kwargs.has_key('sliceobj') and kwargs['sliceobj'] or None
    shape = fseq.shape(subframe)

    tstarts = map(lambda x: int(x[0]/fseq.dt), tranges)
    tstops = map(lambda x: int(x[1]/fseq.dt), tranges)

    out = np.ones((len(tranges),)+shape, np.float64)
    for eds,i,j in cwt_iter(fseq,frange,**kwargs):
        for tk, tr in enumerate(tranges):
            out[tk,i,j] = func(eds[:,tstarts[tk]:tstops[tk]])
    return out

def loc_max_pos(v):
    return [i for i in xrange(1,len(v)-1)
            if (v[i] > v[i-1]) and (v[i] > v[i+1])]

def cwt_freqmap(fseq,
                tranges,
                frange,
                nfreqs = 32,
                **kwargs):
    if len(frange) > 2:
        freqs = frange
    else:
        freqs = np.linspace(frange[0], frange[-1],nfreqs)
    def _dominant_freq(arr):
        ma = np.mean(arr,1) 
        if np.max(ma) < 1e-7:
            print "mean wavelet power %e too low"%np.mean(ma)
            return -1.0
        x = loc_max_pos(ma)
        if x:
            xma = ma[x]
            xma1 = (xma>=np.max(xma)).nonzero()[0]
            n = x[xma1]
        else:
            print "No local maxima. This shouldn't have happened!"
            x = (ma>=np.max(ma)).nonzero()[0]
            try: n = x[0]
            except:
                n = 0
                print x,ma
        return freqs[n]
    return cwtmap(fseq,tranges,freqs,func=_dominant_freq,**kwargs)


def avg_eds(fseq, *args, **kwargs):
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

## def norm1(v, L):
##     return (v - np.mean(v[:L]))/np.std(v[:L])

DoSD = lib.DoSD # Normalization function

def detrend(y, ord=2, take=None):
    x = np.arange(len(y))
    if take is None:
        take = x
    p = np.polyfit(x[take],y[take],ord)
    return y - np.polyval(p, x)


def meanactmap(fseq, (start,stop), normL=None):
    L = fseq.length()
    normL = ifnot(normL, L)
    out = np.zeros(fseq.shape())
    tv = fseq.timevec()
    mrange = (tv > start)*(tv < stop)
    for s,j,k in fseq.pix_iter():
        sx = detrend(s,take=range(330))
        out[j,k] = np.mean(DFoSD(s,normL)[mrange])
    return out
    

def corrmap(fseq, (start, stop), normL=None, sigfunc = tanh_step):
    L = fseq.length()
    normL = ifnot(normL, L)
    out = np.zeros(fseq.shape())
    comp_sig = sigfunc(start, stop)(fseq.timevec())
    for s,j,k in fseq.pix_iter():
        out[j,k] = np.correlate(norm1(s,normL),
                                comp_sig,
                                'valid')[0]
    return out

def fftmap(fseq, frange, func=np.mean,
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

def contiguous_regions_2d(mask):
    """
    Given a binary 2d array, returns a sorted (by size) list of contiguous
    regions (True everywhere)
    TODO: make it possible to use user-defined funtion over an array instead of
    just a binary mask

    """
    import sys
    rl = sys.getrecursionlimit()
    sh = mask.shape
    sys.setrecursionlimit(sh[0]*sh[1])
    regions = []
    rows,cols = mask.shape
    visited = np.zeros(mask.shape, bool)
    for r in xrange(rows):
        for c in xrange(cols):
            if mask[r,c] and not visited[r,c]:
                reg,visited = cont_searcher((r,c), mask, visited)
                regions.append(reg)
    regions.sort(key = lambda x: len(x), reverse=True)

    sys.setrecursionlimit(rl)
    return map(lambda x: Region2D(x,mask.shape), regions)

def filter_proximity(mask, rad=3, size=5, fn = lambda m,i,j: m[i,j]):
    rows, cols = mask.shape
    X,Y = np.meshgrid(xrange(cols), xrange(rows))
    in_circle = lib.in_circle
    out = np.zeros((rows,cols), np.bool)
    for row in xrange(rows):
        for col in xrange(cols):
            if fn(mask,row,col):
                a = in_circle((col,row),rad)
                if np.sum(mask*a(X,Y))>size:
                    out[row,col] = True
    return out

def filter_mask(mask, fn, args=()):
    """Split a mask into contiguous regions, filter their size,
    and return result as a mask
    """
    regs = contiguous_regions_2d(mask)
    filtered_regs = fn(regs, *args)
    z = np.zeros(mask.shape, dtype=np.bool)
    if len(filtered_regs) >1:
        return reduce(lambda a,b:a+b,
                      [z]+[r.tomask() for r in filtered_regs])
    else:
        return z

def filter_size_regions(regions, min_size=5):
    "Filters clusters by their size"
    return [r for r in regions if r.size()>min_size]

def filter_shape_regions(regions, th = 2):
    "Filters continuous regions by their shape"
    return [r for r in regions
            if (r.linsize() > th*np.sqrt(r.size()))]

def cont_searcher(loc, arr, visited):
    """
    Auxilary function for contiguous_regions_2d, finds one contiguous region
    starting from a non-False location
    TODO: make it possible to use user-defined funtion over an array instead of
    just a binary mask
    """
    acc = []
    def _loop(loc, acc):
        if visited[loc]:
            return
        visited[loc] = True
        if arr[loc] and (not loc in acc):
            acc.append(loc)
            for n in neighbours(loc):
                if valid_loc(n, arr.shape):
                    _loop(n,acc)
        else:
            return
    _loop(loc, acc)
    return acc, visited

## def cont_searcher_rec(loc,arr,visited):
##     if arr[loc] and (not loc in acc):
##         visited[loc]=True # side-effect!
##         return [loc] + [cont_searcher_rec(n) for n in neighbours(loc)
##                         if valid_loc(n,arr.shape)]
##     else:
##         return []


def neighbours(loc):
    "list of adjacent locations"
    r,c = loc
    return [(r,c+1),(r,c-1),(r+1,c),(r-1,c),
            (r-1,c-1), (r+1,c-1), (r-1, c+1), (r+1,c+1)]  

def valid_loc(loc,shape):
    "location not outside bounds"
    r,c = loc
    return (0 <= r < shape[0]) and (0<= c < shape[1])

class Region2D:
    "Basic class for a contiguous region. Can make masks from it"
    def __init__(self, locs, shape):
        self.locs = locs
        self.shape = shape
    def size(self,):
        return len(self.locs)
    def center(self):
        return np.mean(self.locs,0)
    def linsize(self,):
        dists = [lib.eu_dist(*pair) for pair in itt.permutations(self.locs,2)]
        return reduce(max, dists)
                               
        pass
    def tomask(self):
        m = np.zeros(self.shape, bool)
        for i,j in self.locs: m[i,j]=True
        return m
            

    
