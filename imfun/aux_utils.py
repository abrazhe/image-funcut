# Auxilary utils for image-funcut

from itertools import combinations


from swan import pycwt
from swan.gui import swancmap

#from pylab import mpl
import pylab as pl
import matplotlib as mpl
import numpy as np

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

def norm1(m1, m2):
    "out=(m1-m2)/m2"
    return (m1-m2)/m2

def ma2d(m, n):
    "Moving average in 2d (for rows)"
    for i in xrange(0,len(m)-n,):
        yield np.mean(m[i:i+n,:],0)


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
                     confidence_level = False):
    wcoefs = pycwt.cwt_f(signal, freqs, f_s, wavelet, padding)
    eds = pycwt.eds(wcoefs, wavelet.f0)
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

def setup_axes1(figsize = (12,6)):
    "Set up axes for a plot with signal, spectrogram and a colorbar"
    fig = pl.figure(figsize = figsize)
    ax = [fig.add_axes((0.08, 0.4, 0.8, 0.5))]
    ax.append(fig.add_axes((0.08, 0.04, 0.8, 0.3), sharex=ax[0]))
    ax.append(fig.add_axes((0.9, 0.4, 0.02, 0.5), 
                           xticklabels=[], 
                           yticklabels=[]))
    return fig,ax



def plot_spectrogram_with_ts(signal, f_s, 
                             lc = 'b', title_string = ''
                             **kwargs):
    "Create a figure of a signal, spectrogram and a colorbar"
    Ns = len(signal)*1.0
    if freqs is None: freqs = default_freqs(Ns, f_s,512)
    tvec = np.arange(0, (Ns+2)/f_s, 1./f_s)[:Ns]

    fig,axlist = setup_axes1()

    axlist[1].plot(tvec, signal,'-',color=lc)
    wavelet_specgram(signal, f_s, freqs,  axlist[0], **kwargs)
    axlist[0].set_title(title_string)
    return fig




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

