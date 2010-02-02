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


def wavelet_specgram(signal, freqs, f_s, ax,
                     wavelet = pycwt.Morlet(),
                     padding = 'zpd',
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
    pl.colorbar(im, ax=ax)
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
    return pl.linspace(4/T, f_s/2, num=num)


import string

class Struct:
    def __init__(self,**kwds):
        self.__dict__.update(kwds)

def lasaf_line_atof(str, sep=';'):
    replacer = lambda s: string.replace(s, ',', '.')
    strlst = map(replacer, str.split(sep))
    return map(np.float, strlst)

def read_lasaf_txt(fname):
    lines = [s.strip() for s in file(fname).readlines()]
    channel = lines[0]
    keys = lines[1].strip().split(';')
    data = np.asarray(map(lasaf_line_atof, lines[2:]))
    dt = data[1:,0]-data[:-1,0]
    j = pl.find(dt>1)[0] + 1
    f_s = 1./np.mean(dt[dt<1])
    return Struct(data=data, jsplit=j, keys = keys, ch=channel, f_s = f_s)

