# Auxilary utils for image-funcut

from swan import pycwt
from swan.gui import swancmap

#from pylab import mpl
import pylab as pl
import matplotlib as mpl
import numpy as np

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