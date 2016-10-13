import numpy as np

import scipy.interpolate as ip
from scipy import ndimage

def rezip(a):
    return zip(*a)


def arr_or(a1,a2):
    return np.vectorize(lambda x,y: x or y)(a1,a2)

def ma2d(m, n):
    "Moving average in 2d (for rows)"
    for i in xrange(0,len(m)-n,):
        yield np.mean(m[i:i+n,:],0)


def __best (scoref, lst):
    if len(lst) > 0:
        n,winner = 0, lst[0]
        for i, item in enumerate(lst):
            if  scoref(item, winner): n, winner = i, item
            return n,winner
    else: return -1,None

def __min1(scoref, lst):
    return __best(lambda x,y: x < y, map(scoref, lst))


def imresize(a, nx, ny, **kw):
    """
    Resize and image or other 2D array with affine transform
    # idea from Sci-Py mailing list (by Pauli Virtanen)
    """
    return ndimage.affine_transform(
        a, [(a.shape[0]-1)*1.0/nx, (a.shape[1]-1)*1.0/ny],
        output_shape=[nx,ny], **kw)

def allpairs(seq):
    return combinations(seq,2)


def allpairs0(seq):
    if len(seq) <= 1: return []
    else:
        return [[seq[0], s] for s in seq[1:]] + allpairs(seq[1:])


def alias_freq(f, fs):
    if f < 0.5*fs:
        return f
    elif 0.5*fs < f < fs:
        return fs - f
    else:
        return alias_freq(f%fs, fs)



def flatten(x,acc=None):
   acc = ifnot(acc,[])
   if not np.iterable(x):
	   acc.append(x)
	   return
   for o in x:
       flatten(o, acc)
   return acc

def ifnot(a, b):
    "if a is not None, return a, else return b"
    if a is None: return b
    else: return a

def som_cluster_fseq(seq, **kwargs):
	from imfun.cluster import som_
	shape = seq.shape()
	a = seq.as3darray()
	tracks = np.array([a[:,i,j] for i,j in
			   itt.product(*map(xrange, shape))])
	perm = np.random.permutation(np.product(shape))
	affiliations = som_.som(tracks,**kwargs)
	return som_.cluster_map_permutation(affiliations, perm, shape)


def vinterpolate(v,n=3,smoothing=1):
    from ..multiscale.atrous import smooth
    L = len(v)
    if smoothing > 0:
        v = smooth(v, smoothing)
    sp2 = ip.UnivariateSpline(np.arange(L),v, s=0)
    xfit = np.linspace(0,L-1,L*n)
    return sp2(xfit)

def ainterpolate(arr, axis=0, n=3, smoothing=1):
    out = None
    #fn = lambda v: vinterpolate(v, n, smoothing)
    def fn(v): return vinterpolate(v, n, smoothing)
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
