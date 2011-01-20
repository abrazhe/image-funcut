# Auxilary utils for image-funcut

from itertools import combinations


from swan import pycwt
from swan.gui import swancmap

#from pylab import mpl
import pylab as pl
import matplotlib as mpl
import numpy as np


def plane(pars, x,y):
	kx,ky,z = pars
        return x*kx + y*ky + z


def remove_plane(arr, pars):
	shape = arr.shape
	X,Y = meshgrid(*map(range,shape))
	return arr - plane(pars, X, Y)

try:
    from scipy import optimize as opt
    from scipy import stats
    def percentile(arr, p):
	    return stats.scoreatpercentile(arr.flatten(), p)
    def fit_plane(arr):
        def _plane_resid(pars, Z, shape):
            Z = reshape(Z,shape)
            X,Y = meshgrid(*map(range,shape))
            return (Z - plane(pars,X,Y)).flatten()
        p0 = pl.randn(3)
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
    "Euler distance between two points"
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

def zero_in_mask(mat, mask):
	out = np.copy(mat)
	out[mask] = 0.0
	return out

def zero_low_sd(mat, n = 1.5):
    return zero_in_mask(mat, mask_median_SD(mat,n,np.less))

def shorten_movie(m,n):
    return np.array([mean(m[i:i+n,:],0) for i in xrange(0, len(m), n)])

def ma2d(m, n):
    "Moving average in 2d (for rows)"
    for i in xrange(0,len(m)-n,):
        yield np.mean(m[i:i+n,:],0)


def flcompose2(f1,f2):
    "Compose two functions from left to right"
    def _(*args,**kwargs):
        return f2(f1(*args,**kwargs))
    return _
                  
def flcompose(*funcs):
    "Compose a list of functions from left to right"
    return reduce(flcompose2, funcs)

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

def best (scoref, lst):
    if len(lst) > 0:
        n,winner = 0, lst[0]
        for i, item in enumerate(lst):
            if  scoref(item, winner): n, winner = i, item
            return n,winner
    else: return -1,None

def min1(scoref, lst):
    return best(lambda x,y: x < y, map(scoref, lst))

def allpairs0(seq):
    return list(combinations(seq,2))

def allpairs(seq):
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


def DoSD(vec, normL=None):
    "Remove mean and normalize to S.D."
    normL = ifnot(normL, len(vec))
    m, sd, x  = np.mean, np.std, vec[:normL]
    return (vec-m(x))/sd(x)

def DFoF(vec, normL=None):
    normL = ifnot(normL, len(vec))
    m = np.mean(vec[:normL])
    if np.all(m):
	return vec/m - 1.0
    else:
	zi = np.where(np.abs(m) < 1e-9)[0]
	m[zi] = 1.0
	out = vec/m - 1.0
	out[zi] = 0
	return out
	
	    
def swanrgb():
    LUTSIZE = mpl.rcParams['image.lut']
    _rgbswan_data =  swancmap.get_rgbswan_data2()
    cmap = mpl.colors.LinearSegmentedColormap('rgbswan',
                                                     _rgbswan_data, LUTSIZE)
    return cmap

def confidence_contour(esurf, extent, ax, L=3.0):
    # Show 95% confidence level (against white noise, v=3 \sigma^2)
    ax.contour(esurf, [L], extent=extent,
               cmap=mpl.cm.gray)

def cone_infl(freqs, extent, wavelet, ax):
    try:
        ax.fill_betweenx(freqs,
                         extent[0],
                         extent[0]+wavelet.cone(freqs),
                         alpha=0.5, color='black')
        ax.fill_betweenx(freqs,
                         extent[1]+wavelet.cone(-freqs),
                         extent[1],
                         alpha=0.5, color='black')
    except:
        print("Can't use fill_betweenx function: update\
        maptlotlib?")


def wavelet_specgram(signal, f_s, freqs,  ax,
                     wavelet = pycwt.Morlet(),
                     padding = 'zpd',
                     cax = None,
                     vmin=None, vmax=None,
                     correct = None,
                     confidence_level = False):
    wcoefs = pycwt.cwt_f(signal, freqs, f_s, wavelet, padding)
    print padding
    eds = pycwt.eds(wcoefs, wavelet.f0)
    if vmax is None: vmax = percentile(eds, 99.0)
    if vmin is None: vmin = percentile(eds, 1.0)
    
    if correct == 'freq1':
        coefs = freqs*2.0/np.pi
        for i in xrange(eds.shape[1]):
            eds[:,i] *= coefs
    endtime = len(signal)/f_s
    extent=[0, endtime, freqs[0], freqs[-1]]
    im = ax.imshow(eds, extent = extent,
                   origin = 'low',
                   vmin = vmin, vmax = vmax,
                   cmap = swanrgb(),
                   alpha = 0.95)
    if not cax:
        pl.colorbar(im, ax=ax)
    else:
        pl.colorbar(im, cax = cax)
    cone_infl(freqs, extent, wavelet, ax)
    if confidence_level:
        confidence_contour(eds, extent, ax, confidence_level)

def group_maps(maplist, ncols,
               titles=None, imkw={}, cbkw ={}):
     import pylab as pl
     nrows = np.ceil(len(maplist)/float(ncols))
     pl.figure(figsize=(2*ncols,2*nrows))
     if not imkw.has_key('aspect'):
	     imkw['aspect'] = 'equal'
     for i,f in enumerate(maplist):
          _ = pl.subplot(nrows,ncols,i+1)
          pl.imshow(f, **imkw);
          pl.colorbar(ax=_,**cbkw);
          if titles is not None: pl.title(titles[i])


def ifnot(a, b):
    "if a is not None, return a, else return b"
    if a == None: return b
    else: return a

def default_freqs(Ns, f_s, num=100):
    """
    Return default frequencies vector
    -- Ns:  number of samples in data vector
    -- f_s: sampling frequency
    -- num: number of frequencies required
    """
    T = Ns/f_s
    return pl.linspace(8/T, f_s/2, num=num)


def alias_freq(f, fs):
    if f < 0.5*fs:
        return f
    elif 0.5*fs < f < fs:
        return fs - f
    else:
        return alias_freq(f%fs, fs)

def setup_axes_for_spectrogram(figsize = (12,6)):
    "Set up axes for a plot with signal, spectrogram and a colorbar"
    fig = pl.figure(figsize = figsize)
    ax = [fig.add_axes((0.08, 0.4, 0.8, 0.5))]
    ax.append(fig.add_axes((0.08, 0.07, 0.8, 0.3), sharex=ax[0]))
    ax.append(fig.add_axes((0.9, 0.4, 0.02, 0.5), 
                           xticklabels=[], 
                           yticklabels=[]))
    return fig,ax



def plot_spectrogram_with_ts(signal, f_s, freqs,
                             figsize=(12,6),
                             lc = 'b', title_string = '',
                             **kwargs):
    "Create a figure of a signal, spectrogram and a colorbar"
    Ns = len(signal)*1.0
    freqs = ifnot(freqs, default_freqs(Ns, f_s,512))
    tvec = np.arange(0, (Ns+2)/f_s, 1./f_s)[:Ns]

    fig,axlist = setup_axes_for_spectrogram(figsize)

    axlist[1].plot(tvec, signal,'-',color=lc)

    kwargs['cax'] = axlist[2]
    wavelet_specgram(signal, f_s, freqs,  axlist[0], **kwargs)
    axlist[0].set_title(title_string)
    axlist[0].axis((tvec[0],tvec[-1], freqs[0],freqs[-1]))
    #axlist[1].xlim((tvec[0],tvec[-1]))
    return fig


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
    except Exception, inst:
        print "%s: Exception"%fname, type(inst)
        return None

