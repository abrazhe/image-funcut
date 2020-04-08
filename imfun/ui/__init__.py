 # a/b will always return float


from .Picker_ import Picker, color_walker
from .plots import group_maps, group_plots, plot_coll

#--- old ui.py ---


import numpy as np

from .. import io
from ..core import ifnot
from .plots import guess_gridshape


from matplotlib import pyplot as plt
from matplotlib import animation
import gc

def harmonize_clims(pickers, mode='extend'):
    lows = [[c[0] for c in px.clims] for px in pickers]
    highs = [[c[1] for c in px.clims] for px in pickers]
    if mode.lower() == 'extend':
        lowfn,highfn = np.amin, np.amax
    elif mode.lower() == 'shrink':
        lowfn,highfn = np.amax, np.amin
    elif mode.lower() == 'mean':
        lowfn,highfn = np.mean,np.mean
    else:
        raise InputError('unknown mode')

    lows = lowfn(np.vstack(lows),0)
    highs = highfn(np.vstack(highs),0)
    return np.array(list(zip(lows, highs)))


codec_ = 'libx264'

def pickers_to_movie(pickers, video_name, fps=25, start=0, stop=None,
                     background = None, frame_pipe=lambda f:f,
                     ncols=None, figsize=None, figscale=4, with_header=True,
                     codec = codec_,
                     titles=None, writer='ffmpeg', bitrate=16000, frame_on=False,
                     marker_idx = None,
                     tight_layout = True,
                     **kwargs):

    plt_interactive = plt.isinteractive()
    plt.ioff() # make animations in non-interactive mode
    if isinstance(pickers, Picker):
        pickers = [pickers]

    marker_idx = ifnot(marker_idx, [])
    stop = ifnot(stop, np.min([len(p.frame_coll) for p in pickers]))
    L = stop-start
    print('number of frames:', L)
    dz = pickers[0].frame_coll.meta['axes'][0] # units of the first frame sequence are used
    zunits = str(dz.unit)
    #dz = dz.value

    #------ Setting canvas up --------------
    if ncols is None:
        nrows, ncols = guess_gridshape(len(pickers))
    else:
        nrows = int(np.ceil(len(pickers)/ncols))
    sh = pickers[0].frame_coll.frame_shape
    aspect = float(sh[0])/sh[1]
    header_add = 0.5
    figsize = ifnot (figsize, (figscale*ncols/aspect, figscale*nrows + header_add))

    fig, axs = plt.subplots(nrows, ncols, figsize=figsize)
    if tight_layout:
        plt.subplots_adjust(left=0.01, right=1-0.01,
                           bottom=0.01, top=0.98,
                           wspace=0.01, hspace=0.02)

    titles = ifnot(titles, ['']*len(pickers))
    if len(titles) < len(pickers):
        titles = list(titles) + ['']*(len(pickers)-len(titles))

    if 'aspect' not in kwargs:
        kwargs['aspect']='equal'
    if 'cmap' not in kwargs:
        kwargs['cmap'] = 'gray'

    views = []


    for p,title,ax in zip(pickers,titles,np.ravel(axs)):

        if 'cmap' in kwargs:
            cmap_= kwargs.pop('cmap')
        else:
            cmap_ = p.cmap

        if background is not None:
            ax.imshow(background)
        view = ax.imshow(frame_pipe(p._get_show_f(start)), vmin=0,vmax=1.0,cmap=cmap_,**kwargs)
        views.append(view)
        ax.set_title(title)
        if not frame_on:
            plt.setp(ax,frame_on=False,xticks=[],yticks=[])
        marker = plt.Rectangle((2,2), 10,10, fc='red',ec='none',visible=False)
        ax.add_patch(marker)
    for ax in np.ravel(axs)[len(pickers):]:
        plt.setp(ax, visible=False)

    header = plt.suptitle('')
    plt.tight_layout()

    # ------ Saving -------------
    def _animate(framecount):
        tstr = ''
        k = start + framecount
        for view, p in zip(views, pickers):
            view.set_data(frame_pipe(p._get_show_f(k)))
        if with_header:
            if zunits in ['sec','msec','s','usec', 'us','ms','seconds']:
                tstr = ', time: %0.3f %s' %(k*dz.value, zunits) #TODO: use in py3 way
            header.set_text('frame %04d'%k + tstr)
        if k in marker_idx:
            plt.setp(marker, visible=True)
        else:
            plt.setp(marker, visible=False)
        return views

    anim = animation.FuncAnimation(fig, _animate, frames=int(L), blit=True)
    #print ("L info: ",L, type(L))
    #anim = animation.FuncAnimation(fig, _animate, frames=1202, blit=True)
    #print ("Save count:", anim.save_count)
    Writer = animation.writers.avail[writer]
    w = Writer(fps=fps,bitrate=bitrate, codec=codec)
    anim.save(video_name, writer=w)

    fig.clf()
    plt.close(fig)
    plt.close('all')
    del anim, w, axs, _animate
    gc.collect()
    if plt_interactive:
        plt.ion()
    return
