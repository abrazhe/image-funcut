# Auxilary utils for image-funcut

from itertools import combinations



#from pylab import mpl
#import pylab as pl
import matplotlib as mpl
import numpy as np

import fnutils

## these functions were in this file, but were moved to fnutils
fnchain = fnutils.fnchain
fniter = fnutils.fniter
flcompose = fnutils.flcompose
take = fnutils.take


import sys
sys.setrecursionlimit(10000)

_dtype_ = np.float64


def rezip(a):
    return zip(*a)

def alist_to_scale(alist):
    names = ("scale", "units")
    formats = ('float', "S10")
    while len(alist) < 3:
        alist.append(alist[-1])
    return np.array(alist, dtype=dict(names=names,formats=formats))


def flatten(x,acc=None):
   acc = ifnot(acc,[])
   if not np.iterable(x):
	   acc.append(x)
	   return 
   for o in x:
       flatten(o, acc)
   return acc

_maxshape_ = 1e9
def memsafe_arr(shape, dtype=_dtype_):
    import tempfile as tmpf
    shape = tuple(shape)
    N = np.prod(shape)
    if N < _maxshape_:
	return np.zeros(shape, dtype=dtype)
    else:
	print "Using memory-mapped arrays..."
	_tmpfile = tmpf.TemporaryFile()
	out = np.memmap(_tmpfile, dtype=dtype, shape=shape)
	_tmpfile.close()
    return out
    

def plane(pars, x,y):
	kx,ky,z = pars
        return x*kx + y*ky + z

def remove_plane(arr, pars):
	shape = arr.shape
	X,Y = np.meshgrid(*map(range,shape[::-1]))
	return arr - plane(pars, X, Y)

try:
    from scipy import optimize as opt
    from scipy.interpolate import splev, splrep
    from scipy import interpolate as ip
    from scipy import stats
    def percentile(arr, p):
	    return stats.scoreatpercentile(arr.flatten(), p)
    def fit_plane(arr):
        def _plane_resid(pars, Z, shape):
            Z = np.reshape(Z,shape)
            X,Y = np.meshgrid(*map(range,shape[::-1]))
            return (Z - plane(pars,X,Y)).flatten()
        p0 = np.random.randn(3)
	p1 = opt.leastsq(_plane_resid, p0, (arr.flatten(), arr.shape))[0]
        return p1
except:
    _scipyp = False
    
def vessel_mask(f, p, negmask, thresh = None):
	sh = f.shape
	X,Y = meshgrid(*map(range, sh))
	fx = f - plane(p,X,Y)
	if thresh is None:
		posmask = fx > median(fx) + fx.std()
	else:
		posmask = fx > thresh
	return posmask*negmask
	
def in_circle(coords, radius):
    return lambda x,y: (square_distance((x,y), coords) <= radius**2)

def eu_dist(p1,p2):
    return np.sqrt(np.sum([(x-y)**2 for x,y in zip(p1,p2)]))

def eu_dist2d(p1,p2):
    "Euclidean distance between two points"
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def square_distance(p1,p2):
    return (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2

def mask_percent_threshold(mat, thresh):
    minv = np.min(mat)
    maxv = np.max(mat)
    val = (maxv-minv) * thresh/100.0
    return mat > val

def mask_threshold(mat, thresh, func=lambda a,b: a>b):
    return mat > thresh

def mask_num_std(mat, n, func=lambda a,b: a>b):
    "Same as threshold, but threshold value is times S.D. of the matrix"
    x = np.std(mat)
    return func(mat, x*n)

def mask_median_SD(mat, n = 1.5, compfn = np.greater):
    return compfn(mat, np.median(mat) + n*mat.std())

def mask_low_percentile(mat, threshold = 15.0):
    low = np.percentile(np.ravel(mat), threshold)
    return mat < low

def invert_mask(m):
    def _neg(a):
        return not a
    return np.vectorize(_neg)(m)

def zero_in_mask(mat, mask):
	out = np.copy(mat)
	out[mask] = 0.0
	return out

def zero_low_sd(mat, n = 1.5):
    return zero_in_mask(mat, mask_median_SD(mat,n,np.less))

def arr_or(a1,a2):
    return np.vectorize(lambda x,y: x or y)(a1,a2)

def shorten_movie(m,n):
    return np.array([mean(m[i:i+n,:],0) for i in xrange(0, len(m), n)])

def ma2d(m, n):
    "Moving average in 2d (for rows)"
    for i in xrange(0,len(m)-n,):
        yield np.mean(m[i:i+n,:],0)



def with_time_dec(fn):
    "decorator to time function evaluation"
    def _(*args, **kwargs):
	import time
	t = time.time()
	out = fn(*args,**kwargs)
        print "time lapsed %03.3e in %s"%(time.time() - t, str(fn))
	return out
    _.__doc__ = fn.__doc__
    return _

## not needed, use ipython magic %time
def with_time(fn, *args, **kwargs):
    "take a function and timer its evaluation"
    import time
    t = time.time()
    out = fn(*args,**kwargs)
    print "time lapsed %03.3e in %s"%(time.time() - t, str(fn))
    return out
    

def ensure_dir(f):
	import os
	d = os.path.dirname(f)
	if not os.path.exists(d):
		os.makedirs(d)
	return f


def imresize(a, nx, ny, **kw):
    """
    Resize and image or other 2D array with affine transform
    # idea from Sci-Py mailing list (by Pauli Virtanen)
    """
    from scipy import ndimage
    return ndimage.affine_transform(
        a, [(a.shape[0]-1)*1.0/nx, (a.shape[1]-1)*1.0/ny],
        output_shape=[nx,ny], **kw) 

def __best (scoref, lst):
    if len(lst) > 0:
        n,winner = 0, lst[0]
        for i, item in enumerate(lst):
            if  scoref(item, winner): n, winner = i, item
            return n,winner
    else: return -1,None

def __min1(scoref, lst):
    return best(lambda x,y: x < y, map(scoref, lst))

def allpairs(seq):
    return combinations(seq,2)

def allpairs0(seq):
    if len(seq) <= 1: return []
    else:
        return [[seq[0], s] for s in seq[1:]] + allpairs(seq[1:])

def ar1(alpha = 0.74):
    "Simple auto-regression model"
    randn = np.random.randn
    prev = randn()
    while True:
        res = prev*alpha + randn()
        prev = res
        yield res

def ifnot(a, b):
    "if a is not None, return a, else return b"
    if a == None: return b
    else: return a

def embedding(arr, delarrp=True):
    """Return an *embeding* of the non-zero portion of an array.

    Parameters:
      - `arr`: array
      - `delarrp`: predicate whether to delete the `arr`

    Returns tuple ``(out, (sh, slices))`` of:
        * out: array, which is a bounding box around non-zero elements of an input
          array
	* sh:  full shape of the input data
	* slices: a list of slices which define the bounding box
    
    
    """
    sh = arr.shape
    b = np.argwhere(arr)
    starts, stops = b.min(0), b.max(0)+1
    slices = [slice(*p) for p in zip(starts, stops)]
    out =  arr[slices].copy()
    if delarrp: del arr
    return out, (sh, slices)

def embedded_to_full(x):
    """Restore 'full' object from it's *embedding*, 
    e.g. full image from  object subframe
    """
    data, (shape, xslice) = x
    out = np.zeros(shape, _dtype_)
    out[xslice] = data
    return out
    

def auto_threshold(arr, init_th = None, max_iter = 1e7):
    """
    Automatic threhold with INTERMEANS(I) algorithm

    Parameters:
      - `arr`: array-like
      - `init_th`: starting threshold
      - `max_iter`: upper limit of iterations

    Returns:
      - threshold: float

    Based on:
    T. Ridler and S. Calvard, "Picture thresholding using an iterative
    selection method," IEEE Trans. Systems Man Cybernet., vol. 8, pp. 630-632,
    1978.
    """
    thprev = ifnot(init_th, np.median(arr))
    for i in xrange(int(max_iter)):
	ab = np.mean(arr[np.where(arr <= thprev)])
	av = np.mean(arr[np.where(arr > thprev)])
	thnext = 0.5*(ab+av)
	if thnext <= thprev:
		break
	thprev = thnext
    return thnext

def DFoSD(vec, normL=None, th = 1e-6):
    "Subtract mean value along first axis and normalize to S.D."
    normL = ifnot(normL, len(vec))
    m, x  = np.mean, vec[:normL]
    sdx = np.std(x,0)
    out =  np.zeros(vec.shape, vec.dtype)
    if sdx.shape is ():
	if np.abs(sdx) > th:
		out = (vec-m(x))/sdx
    else:
	zi = np.where(np.abs(sdx) < th)[0]
	sdx[zi] = -np.inf
	out = (vec-m(x))/sdx
    return out

def DFoF(vec, normL=None, th = 1e-6):
    "Subtract mean value along first axis and normalize to it"
    normL = ifnot(normL, len(vec))
    m = np.mean(vec[:normL], 0)
    out = np.zeros(vec.shape, vec.dtype)
    if m.shape is ():
	if np.abs(m) > th:
	    out =  vec/m - 1.0
    else:
	zi = np.where(np.abs(m) < th)
	m[zi] = -np.inf
	out = vec/m - 1.0
    return out


def clip_and_rescale(arr,nout=100):
    "convert data to floats in 0...1, throwing out nout max values"
    out = arr - np.min(arr)
    cutoff = 100*(1-float(nout)/np.prod(arr.shape))
    m = np.percentile(arr, cutoff)
    return np.where(arr < m, arr, m)/m

def rescale(arr):
    "Rescales array to [0..1] interval"
    out = arr - np.min(arr)
    return out/np.max(out)
	

def mask4overlay(mask,colorind=0, alpha=0.9):
    """
    Put a binary mask in some color channel
    and make regions where the mask is False transparent
    """
    sh = mask.shape
    z = np.zeros(sh)
    stack = np.dstack((z,z,z,alpha*np.ones(sh)*mask))
    stack[:,:,colorind] = mask
    return stack

def mask4overlay2(mask,color=(1,0,0), alpha=0.9):
    """
    Put a binary mask in some color channel
    and make regions where the mask is False transparent
    """
    sh = mask.shape
    ch = lambda i: np.where(mask, color[i],0)
    stack = np.dstack((ch(0),ch(1),ch(2),alpha*np.ones(sh)*mask))
    return stack


from scipy import sparse
#spsolve = sparse.linalg.spsolve
def baseline_als(y, lam=None, p=0.1, niter=10):
    """Implements an Asymmetric Least Squares Smoothing
    baseline correction algorithm
    (P. Eilers, H. Boelens 2005)
    """
    L = len(y)
    if lam == None:
	lam = L**2
    D = sparse.csc_matrix(np.diff(np.eye(L),2))
    w = np.ones(L)
    for i in xrange(niter):
	W = sparse.spdiags(w, 0, L, L)
	Z = W + lam*D.dot(D.T)
	z = sparse.linalg.spsolve(Z,w*y)
	w = p*(y>z) + (1-p)*(y<z)
    return z


def locextr(v, x=None, refine = True, output='full',
	    sort_values = True,
	    **kwargs):
       "Finds local extrema "
       if x is None: x = np.arange(len(v))
       tck = splrep(x,v, **kwargs) # spline representation
       if refine:
               xfit = np.linspace(x[0],x[-1], len(x)*10)
       else:
               xfit = x
       yfit = splev(xfit, tck)
       der1 = splev(xfit, tck, der=1)
       #der2 = splev(xfit, tck, der=2)
       dersign = np.sign(der1)

       maxima = np.where(np.diff(dersign) < 0)[0]
       minima = np.where(np.diff(dersign) > 0)[0]
       if sort_values:
           maxima = sorted(maxima, key = lambda p: yfit[p], reverse=True)
           minima = sorted(minima, key = lambda p: yfit[p], reverse=False)
       if output=='full':
           return xfit, yfit, der1, maxima, minima 
       elif output=='max':
           return zip(xfit[maxima], yfit[maxima])
       elif output =='min':
           return zip(xfit[minima], yfit[minima])
	

def extrema2(v, *args, **kwargs):
   "First and second order extrema"
   xfit,yfit,der1,maxima,minima = locextr(v, *args, **kwargs)
   xfit, _, der2, gups, gdowns = locextr(der1, x=xfit, refine=False)
   return (xfit, yfit), (maxima, minima), (gups, gdowns)

def vinterpolate(v,n=3,smoothing=1):
    from atrous import smooth
    L = len(v)
    if smoothing > 0:
        v = smooth(v, smoothing)
    sp2 = ip.UnivariateSpline(np.arange(L),v, s=0)
    xfit = np.linspace(0,L-1,L*n)
    return sp2(xfit)

def ainterpolate(arr, axis=0, n=3, smoothing=1):
    out = None
    fn = lambda v: vinterpolate(v, n, smoothing)
    if axis == 1:
        out = np.array(map(fn, arr))
    elif axis ==0:
        out = np.array(map(fn, arr.T)).T
    else:
        print "Can't work with these many dimensions yet"
    return out

def simple_snr(v,plow=50,phigh=75):
    ml,mu = np.percentile(v, (plow, phigh))
    return np.mean(v[v>mu])/(np.std(v[v<=ml])+1.0)

def simple_snr2(arr, plow=50,phigh=75):
    nrows, ncols = arr.shape
    out = np.zeros(ncols)
    for j in xrange(ncols):
        c1 = np.abs(simple_snr(arr[:,j],plow,phigh))
        c2 = np.abs(simple_snr(-arr[:,j],plow,phigh))
        out[j] = max(c1,c2)
    out /= out.mean()
    return out

def group_maps(maplist, ncols=None,
               titles=None,
               figscale = 2,
	       figsize = None,
	       suptitle = None,
	       background = None,
	       individual_colorbars = False,
	       single_colorbar = None,
	       show_ticks = False,
	       samerange = True,
	       imkw=None, cbkw ={}):
    import pylab as pl
    if imkw is None:
        imkw = {}
    else:
        imkw = imkw.copy()
    if ncols is None:
	ncols = min(10, len(maplist))
    nrows = int(np.ceil(len(maplist)/float(ncols)))
    sh = maplist[0].shape
    aspect = float(sh[0])/sh[1]
    figsize = ifnot (figsize, (figscale*ncols/aspect,figscale*nrows)) 
    figh = pl.figure(figsize=figsize)
    print samerange
    if samerange:
	vmin,vmax = data_range(maplist)
	imkw.update(dict(vmin=vmin, vmax=vmax))
	if single_colorbar is None:
	    single_colorbar = True
    else:
	if single_colorbar is None:
	    single_colorbar=False
    if not imkw.has_key('aspect'):
	imkw['aspect'] = 'equal'
    for i,f in enumerate(maplist):
	ax = pl.subplot(nrows,ncols,i+1)
	if background is not None:
	    ax.imshow(background, cmap='gray', aspect='equal')
	im = ax.imshow(f, **imkw);
	if not show_ticks:
	    pl.setp(ax, 'xticks', [], 'yticks', [],
		    'frame_on', False)
	if individual_colorbars:
	    figh.colorbar(im, ax=ax);
	if titles is not None: pl.title(titles[i])
    if single_colorbar:
	pl.subplots_adjust(bottom=0.1, top=0.9, right=0.8)
	cax = pl.axes([0.85, 0.1, 0.03, 0.618])
	pl.colorbar(im, cax=cax, **cbkw)
    if suptitle:
        pl.suptitle(suptitle)
    return

def data_range(datalist):
   vmin = np.min(map(np.min, datalist))
   vmax = np.max(map(np.max, datalist))
   return vmin, vmax

def group_plots(ylist, ncols, x = None,
		titles = None,
		suptitle = None,
		ylabels = None,
		figsize = None,
		sameyscale = True,
                order='C',
		imkw={}):
    import pylab as pl
    nrows = np.ceil(len(ylist)/float(ncols))
    figsize = ifnot(figsize, (2*ncols,2*nrows))
    fh, axs = pl.subplots(int(nrows), int(ncols),
                          sharex=True,
                          sharey=bool(sameyscale),
                          figsize=figsize)
    ymin,ymax = data_range(ylist)
    axlist = axs.ravel(order=order)
    for i,f in enumerate(ylist):
	x1 = ifnot(x, range(len(f)))
        _im = axlist[i].plot(x1,f,**imkw)
	if titles is not None:
            pl.setp(axlist[i], title = titles[i])
	if ylabels is not None:
            pl.setp(axlist[i], ylabel=ylabels[i])
    if suptitle:
        pl.suptitle(suptitle)
    return

	
###------------- Wavelet-related -------------	    

def alias_freq(f, fs):
    if f < 0.5*fs:
        return f
    elif 0.5*fs < f < fs:
        return fs - f
    else:
        return alias_freq(f%fs, fs)

###---------- End Wavelet-related -------------	


### Stackless trampolining
def trampoline(function, *args):
    """Bounces a function over and over, until we "land" off the
    trampoline."""
    bouncer = bounce(function, *args)
    while True:
        bouncer = bouncer[1](*bouncer[2])
        if bouncer[0] == 'land':
            return bouncer[1]


def bounce(function, *args):
    """Bounce back onto the trampoline, with an upcoming function call."""
    return ["bounce", function, args]


def land(value):
    """Jump off the trampoline, and land with a value."""
    return ["land", value]
### --- end of stackless trampolining


## This is for reading Leica txt files
## todo: move to readleicaxml?
import string

class Struct:
    def __init__(self,**kwds):
        self.__dict__.update(kwds)

def lasaf_line_atof(str, sep=';'):
    replacer = lambda s: string.replace(s, ',', '.')
    strlst = map(replacer, str.split(sep))
    return map(np.float, strlst)

def read_lasaf_txt(fname):
    try:
        lines = [s.strip() for s in file(fname).readlines()]
        channel = lines[0]
        keys = lines[1].strip().split(';')
        data = np.asarray(map(lasaf_line_atof, lines[2:]))
        dt = data[1:,0]-data[:-1,0]
        j = pl.find(dt>=max(dt))[0] + 1
        f_s = 1./np.mean(dt[dt<max(dt)])
        return Struct(data=data, jsplit=j, keys = keys, ch=channel, f_s = f_s)
    except Exception as inst:
        print "%s: Exception"%fname, type(inst)
        return None


### Monte-Carlo routines for significance level estimation

 
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
	s0 = ','.join(['%1.2e'%a for a in x])
	s = '\r signal {:06d} out of {:06d}, current: {}'.format(n+1,long(N), s0)
	sys.stderr.write(s)
	#out[n] = map(np.std, decompose1d_direct(im, level)[:-1])
        out[n] = map(np.std, decompose_fn(im, level)[:-1])
    return np.mean(out, axis=0)


def som_cluster_fseq(seq, **kwargs):
	import itertools as itt
	from imfun import som
	shape = seq.shape()
	a = seq.as3darray()
	tracks = np.array([a[:,i,j] for i,j in
			   itt.product(*map(xrange, shape))])
	perm = np.random.permutation(np.product(shape))
	affiliations = som.som1(tracks,**kwargs)
	return som.cluster_map_permutation(affiliations, perm, shape)
	
def n_random_locs(n, shape):
    """
    return a list of n random locations within shape
    """
    return zip(*[tuple(np.random.choice(dim, n)) for dim in shape])


def write_table_csv(table, fname):
    import csv
    with open(fname, 'w') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        for row in table:
            writer.writerow(row)

def write_dict_csv(dict, fname):
    import csv
    with open(fname, 'w') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        keys = sorted(dict.keys())
        # make all data entries one-dimensional (surprize!)
        vals = [np.ravel(dict[k]) for k in keys]
        Lmax = np.max([len(v) for v in vals])
        writer.writerow([''] + keys)
        for i in xrange(Lmax):
            row = [i] + [v[i] if len(v) >i and v[i] is not None and not np.isnan(v[i]) else 'NA' for v in vals]
            writer.writerow(row)
        
        
        
