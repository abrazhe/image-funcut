"""
Helper functions for some special plot kinds
"""



import numpy as np

import matplotlib as mpl
from matplotlib import pyplot as pl

from ..core import ifnot


_gridshapes = {4:(2,2),
               5:(2,3),
               6:(2,3),
               7:(2,4),
               8:(2,4),
               9:(3,3),
               10:(2,5),
               11:(3,4),
               12:(3,4),
               13:(2,7),
               14:(2,7),
               15:(3,5),
               16:(4,4),
               17:(3,6),
               18:(3,6),
               21:(3,7),}

def guess_gridshape(nelements):
    if nelements in _gridshapes:
        nrows, ncols = _gridshapes[nelements]
    else:
        ncols= min(10, nelements)
        nrows = int(np.ceil(nelements/ncols))
    return nrows, ncols


#todo: convert to using subplots
def group_maps(maplist, ncols=None,
               titles=None,
               figscale = 2,
               figsize = None,
               suptitle = None,
               background = None,
               individual_colorbars = False,
               colorbar = None,
               show_ticks = False,
               samerange = True,
               imkw=None, cbkw ={}):
    import pylab as pl
    if imkw is None:
        imkw = {}
    else:
        imkw = imkw.copy()

    if ncols is None:
        nrows, ncols = guess_gridshape(len(maplist))
    else:
        nrows = int(np.ceil(len(maplist)/ncols))
    sh = maplist[0].shape
    aspect = float(sh[0])/sh[1]
    figsize = ifnot (figsize, (figscale*ncols/aspect,figscale*nrows))
    figh = pl.figure(figsize=figsize)
    #print samerange
    if samerange:
        vmin,vmax = data_range(maplist)
        imkw.update(dict(vmin=vmin, vmax=vmax))
        if colorbar is None:
            colorbar = True
    else:
        if colorbar is None:
            colorbar=False
    if 'aspect' not in imkw:
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
    if colorbar:
        pl.subplots_adjust(bottom=0.1, top=0.9, right=0.8)
        cax = pl.axes([0.85, 0.1, 0.03, 0.618])
        pl.colorbar(im, cax=cax, **cbkw)
    if suptitle:
        pl.suptitle(suptitle)
    return

def data_range(datalist):
   vmin = np.min(list(map(np.min, datalist)))
   vmax = np.max(list(map(np.max, datalist)))
   return vmin, vmax


def plot_coll(vecs,x=None,sep=None,positions=None,colors=None,
              ax = None,
              figsize=None,
              frame_on=False,
              labels = None,
              **kwargs):
    

    if sep is None:
        mean_range = np.mean([np.max(v)-np.min(v) for v in vecs])
        sep = 0.05*mean_range

    if colors is None: colors = 'b'
    if labels is None: labels = [None]*len(vecs)
    if isinstance(colors, str):
        c = colors
        colors = (c for i in range(int(1e6)))
    if positions is None:
        prevpos,positions = 0,[0]
        ranges = [(v.min(),v.max()) for v in vecs]
        for r,rnext,v in zip(ranges, ranges[1:],vecs):
            pos = prevpos + r[1] + sep -np.min(rnext[0])
            positions.append(pos)
            prevpos = pos
    L = np.min(list(map(len, vecs)))
    if x is None:
        x = np.arange(L)
    if ax is None:
        f,ax = pl.subplots(1,1,figsize=figsize)
    for v,p,c,l in zip(vecs,positions[::-1],colors,labels):
        ax.plot(x, v[:L]+p, color=c, label=l, **kwargs)
        #a.axhline(p, color='b')
    pl.setp(ax, yticks=[],frame_on=frame_on)
    ax.axis('tight')

def group_plots(ylist, ncols=None, x = None,
                titles = None,
                suptitle = None,
                ylabels = None,
                figsize = None,
                sameyscale = True,
                order='C',
                imkw={}):
    import pylab as pl

    if ncols is None:
        nrows, ncols = guess_gridshape(len(ylist))
    else:
        nrows = int(np.ceil(len(ylist)/ncols))

    figsize = ifnot(figsize, (2*ncols,2*nrows))
    fh, axs = pl.subplots(int(nrows), int(ncols),
                          sharex=True,
                          sharey=bool(sameyscale),
                          figsize=figsize)
    ymin,ymax = data_range(ylist)
    axlist = axs.ravel(order=order)
    for i,f in enumerate(ylist):
        x1 = ifnot(x, list(range(len(f))))
        _im = axlist[i].plot(x1,f,**imkw)
        if titles is not None:
            pl.setp(axlist[i], title = titles[i])
        if ylabels is not None:
            pl.setp(axlist[i], ylabel=ylabels[i])
    if suptitle:
        pl.suptitle(suptitle)
    return

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
    #ch = lambda i: np.where(mask, color[i],0)
    def ch(i): return np.where(mask, color[i],0)
    stack = np.dstack((ch(0),ch(1),ch(2),alpha*np.ones(sh)*mask))
    return stack
