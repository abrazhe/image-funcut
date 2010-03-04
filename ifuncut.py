#!/usr/bin/python

import os
import sys
import glob
import time
import pylab as pl
from itertools import combinations

import numpy as np
from pylab import *
from pylab import mpl

from swan import pycwt
from imfun import aux_utils as aux
from imfun.aux_utils import ifnot

from scipy import signal



mpl.rcParams['image.aspect'] = 'auto'
mpl.rcParams['image.origin'] = 'lower'



def nearest_item_ind(items, xy, fn = lambda a: a):
    "Index of nearest item from collection. Arguments: collection, position, selector"
    return aux.min1(lambda p: eu_dist(fn(p), xy), items)



def eu_dist(p1,p2):
    return sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def square_distance(p1,p2):
    return (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2


def line_from_points(p1,p2):
    p1 = map(float, p1)
    p2 = map(float, p2)
    k = (p1[1] - p2[1])/(p1[0] - p2[0])
    b = p1[1] - p1[0]*k
    return lambda x: k*x + b

def in_circle(coords, radius):
    return lambda x,y: square_distance((x,y), coords) < radius**2


def extract_line(data, xind, f):
    return array([data[int(i),int(f(i))] for i in xind])


def extract_line2(data, p1, p2):
    L = int(eu_dist(p1,p2))
    f = lambda x1,x2: lambda i: int(x1 + i*(x2-x1)/L)
    return array([data[f(p1[1],p2[1])(i), f(p1[0],p2[0])(i)] for i in range(L)])


def percent_threshold(mat, thresh,
                      func = lambda a,b: a>b):
    minv = np.min(mat)
    maxv = np.max(mat)
    val = (maxv-minv) * thresh/100.0
    return func(mat,val)

def threshold(mat, thresh, func=lambda a,b: a>b):
    """
    Given a matrix M, a value V and a function f, return a matrix X of Booleans
    where each element is f M(i,j) v -> X(i,j).

    By default, `f a b -> True if a > b, False otherwise`
    i.e. `lambda a,b: a>b`
    """
    return func(mat, thresh)

def times_std(mat, n, func=lambda a,b: a>b):
    "Same as threshold, but threshold value is times S.D. of the matrix"
    x = np.std(mat)
    return func(mat, x*n)


def file_names(pat):
    "Returns a sorted list of file names matching a pattern"
    x = glob.glob(pat)
    x.sort()
    return x

class ImageSequence:
    "Base Class for image sequence"
    def __init__(self, pattern, ch=0, dt = 0.16):
        """
        Creates an instance. Image files must be in RGB format
        - pattern: a string like "/path/to/files/*.tif"
        - ch: an integer, 0--2, sets default channel. 0 is red, 1 is green
        and 2 is blue.
        - dt: a float, specifies time delay between the frames
        """
        self.ch = ch
        self.dt = dt
        #self.d = array([imread(f) for f in file_names(pattern)])
        self.d = [imread(f) for f in file_names(pattern)]
        self.Nf = len(self.d)
        self.shape = self.d[0].shape[:-1]

    def ch_view(self, ch=None):
        """
        Returns values for a given color channel in the image sequence (stack)
        as a 3D array.
        """
        ch = ifnot(ch, self.ch)
#        return self.d[:,:,:,ch]
        return array([d[:,:,ch] for d in self.d])

    def bmask_timeseries(self, mask, ch=None):
        """
        Given a binary mask (matrix of True or False values, with dimensions equal
        to the dimensions of frames), return 1D timeseries, where each value is
        an averaged value for each freame where mask is True.
        """
        ch = ifnot(ch, self.ch)
        return array([mean(m[mask,ch]) for m in self.d])

    def default_kernel(self):
        """
        Default kernel for aliased_pix_iter2
        Used in 2D convolution of each frame in the sequence
        """
        kern = ones((3,3))
        kern[1,1] = 2.0
        return kern/sum(kern)

    def aliased_pix_iter2(self, ch=None, kern=None):
        #arr = self.ch_view(ch)
        ch = ifnot(ch, self.ch)
        nrows,ncols = self.shape

        kern = ifnot(kern, self.default_kernel())
        
        arr = asarray([signal.convolve2d(m[:,:,ch], kern)[1:-1,1:-1]
                       for m in self.d ])
        for i in range(nrows):
            for j in range(ncols):
                yield arr[:,i,j], i, j


    def simple_pix_iter(self, ch=None):
        arr = self.ch_view(ch)
        nrows,ncols = self.shape
        for i in range(nrows):
            for j in range(ncols):
                yield arr[:,i,j], i, j


    def get_time(self):
        """
        Return a vector of time values, calculated from sampling interval and
        number of frames
        """
        return arange(0,self.Nf*self.dt, self.dt)[:self.Nf]


    def mean(self, ch=None):
        """
        Returns 'mean frame' along the sequence in a given color channel as a
        2D numpy array
        """
        ch = ifnot(ch, self.ch)
        return np.mean(self.ch_view(ch), axis=0)

    def frame_view(self, ch=None):
        """
        Shows the data frame by frame
        """
        ch = ifnot(ch, self.ch)
        self.figf = figure()
        self.axf = axes()
        self.plf = self.axf.imshow(self.d[0][:,:,ch], origin='low',
                                   aspect = 'equal', cmap=mpl.cm.gray)
        self.frame_index = 0
        def onscroll(event):
            if event.button in ('4','down'):
                self.frame_index -= 1
            elif event.button in ('5','up'):
                self.frame_index += 1
            self.frame_index = self.frame_index%self.Nf
            ind = self.frame_index
            self.plf.set_data(self.d[ind][:,:,ch])
            self.axf.set_xlabel('%03d'%ind)
            draw()

        self.figf.canvas.mpl_connect('scroll_event', onscroll)
        

    def show(self, ch=None):
        """
        Shows a mean frame along the sequence
        """
        ch = ifnot(ch, self.ch)
        self.fig = figure()
        self.ax = axes()
        self.pl = imshow(self.mean(ch), origin='low',
                         aspect='equal', cmap=matplotlib.cm.gray)

def rezip(a):
    return zip(*a)

def shorten_movie(m,n):
    return array([mean(m[i:i+n,:],0) for i in xrange(0, len(m), n)])


class ImgLineSect(ImageSequence):
    verbose = True
    def check_endpoints(self):
        if self.endpoints[0][0] > self.endpoints[1][0]:
            return self.endpoints[::-1]
        else:
            return self.endpoints

    def ontype_timeser(self, event):
        if event.inaxes == self.ax2:
            if event.key == 'pageup':
                self.nshort += 1
            elif event.key == 'pagedown':
                self.nshort -= 1
                if self.nshort < 1: self.nshort = 1
            if self.pl2 is not None:
                setp(self.pl2, 'data', self.prepare_timeseries())

    def onclick(self, event):
        tb = get_current_fig_manager().toolbar
        if event.inaxes == self.ax1 and tb.mode == '':
            x,y = round(event.xdata), round(event.ydata)
            if self.verbose:
                print "You clicked:", x,y
            if event.button == 1:
                self.endpoints[0] = (x,y)
            else:
                self.endpoints[1] = (x,y)
            if not None in self.endpoints:    
                if self.pl_seg is None:
                    self.pl_seg = self.ax1.plot(*rezip(self.endpoints),
                                                linestyle='-',
                                                marker='.',
                                                color='red')
                else:
                    setp(self.pl_seg, 'data', rezip(self.endpoints))
                axis([0, self.shape[0], 0, self.shape[1]])
                self.show_timeseries()
            self.fig.canvas.draw()

    def binarize_scroller(self, event):
        step = -self.bin_step
        if event.inaxes == self.ax2:
            if event.button in ['up', 1]:
                step = self.bin_step
        self.bin_thresh += step
        if self.pl2 is not None:
            setp(self.pl2, 'data', self.prepare_timeseries())

    def prepare_timeseries(self):
        return self.binarizer(shorten_movie(self.make_timeseries(), self.nshort))
    
    def binarizer(self, im):
        #return im
        return im*percent_threshold(im, self.bin_thresh)
    
    def select(self, ch=None):
        self.bin_thresh = 20
        self.fig = figure()
        self.ax2 = subplot(122)
        self.ax1 = subplot(121); hold(True)
        self.pl = self.ax1.imshow(self.mean(ch),
                                  aspect='equal',
                                  origin = 'low',
                                  cmap=matplotlib.cm.gray)
        self.pl2 = None
        self.pl_seg = None
        self.endpoints = [None, None]
        self.fig.canvas.mpl_connect('button_press_event',
                                    self.onclick)
        self.fig.canvas.mpl_connect('scroll_event',
                                    self.binarize_scroller)
        self.fig.canvas.mpl_connect('key_press_event',
                                    self.ontype_timeser)


    def make_timeseries(self, ch=None):
        points = self.check_endpoints()
        return  array([extract_line2(d,*points) for d in self.ch_view(ch)])

    def show_timeseries(self):
        self.ax2.cla()
        self.bin_step = 0.5
        self.nshort = 1
        self.pl2 = self.ax2.imshow(
            self.prepare_timeseries(),
            origin='low',
            extent = (0, int(eu_dist(*self.endpoints)),
                      0, self.Nf*self.dt),
            interpolation='nearest')
        
    def get_timeview(self):
        return self.make_timeseries()

import itertools

def ar1(alpha = 0.74):
    prev = randn()
    while True:
        res = prev*alpha + randn()
        prev = res
        yield res

def color_walker():
    red, green, blue = ar1(), ar1(), ar1()
    while True:
        yield map(lambda x: mod(x.next(),1.0), (red,green,blue))

def struct_circle(circ):
    return {'center': circ.center,
            'radius': circ.radius,
            'alpha': circ.get_alpha(),
            'label': circ.get_label(),
            'color': circ.get_facecolor()}

def circle_from_struct(circ_props):
    return Circle(circ_props['center'], circ_props['radius'],
                  alpha=circ_props['alpha'],
                  label = circ_props['label'],
                  color = circ_props['color'])
    pass

import pickle
import random

vowels = "aeiouy"
consonants = "qwrtpsdfghjklzxcvbnm"


def rand_tag():
    return ''.join((random.choice(consonants),
                    random.choice(vowels),
                    random.choice(consonants)))

def unique_tag(tags, max_tries = 1e4):
    n = 0
    while n < max_tries:
        tag = rand_tag()
        if not tag in tags:
            return tag
    return "Err"


class DraggableCircle():
    verbose = True
    "Draggable Circle ROI"
    def __init__ (self, circ, parent = None):
        self.circ = circ
        self.parent = parent
        self.press = None
        self.connect()

    def connect(self):
        "connect all the needed events"
        self.cidpress = \
        self.circ.figure.canvas.mpl_connect('button_press_event',
                                            self.on_press)
        self.cidrelease = \
        self.circ.figure.canvas.mpl_connect('button_release_event',
                                            self.on_release)

        self.cidscroll = \
        self.circ.figure.canvas.mpl_connect('scroll_event',
                                            self.on_scroll)

        self.cidmotion = \
        self.circ.figure.canvas.mpl_connect('motion_notify_event',
                                            self.on_motion)
        self.cidtype = \
        self.circ.figure.canvas.mpl_connect('key_press_event',
                                            self.on_type)             

    def disconnect(self):
        "Disconnect stored connections"
        map(self.circ.figure.canvas.mpl_disconnect,
            (self.cidpress, self.cidrelease, self.cidscroll,
             self.cidmotion, self.cidtype))

    def on_scroll(self, event):
        if event.inaxes != self.circ.axes : return
        contains, attrd = self.circ.contains(event)
        if not contains : return 
        step = 1
        r = self.circ.get_radius()
        if event.button in ['up']:
            self.circ.set_radius(r+step)
        else:
            self.circ.set_radius(max(0.1,r-step))
        self.circ.figure.canvas.draw()
        return

    def on_type(self, event):
        if event.inaxes != self.circ.axes: return
        contains, attrd = self.circ.contains(event)
        if not contains: return
        if self.verbose:
            print event.key
        tags = [self.circ.get_label()]
        if event.key in ['t', '1']:
            self.parent.show_timeseries(tags)
        if event.key in ['T', '!']:
            self.parent.show_timeseries(tags, normp=True)
        elif event.key in ['w', '2']:
            self.parent.show_spectrograms(tags)
        elif event.key in ['W', '3']:
            self.parent.show_wmps(tags)
        elif event.key in ['4']:
            self.parent.show_ffts(tags)
                                        


    def on_press(self, event):
        if event.inaxes != self.circ.axes: return
        contains, attrd = self.circ.contains(event)
        if not contains: return
        x0,y0 = self.circ.center
        if event.button is 1:
            self.press = x0, y0, event.xdata, event.ydata
        elif event.button is 2:
            self.parent.show_timeseries([self.circ.get_label()])
            
        elif event.button is 3:
            p = self.parent.drcs.pop(self.circ.get_label())
            self.circ.remove()
            self.disconnect()
            legend()
            self.circ.figure.canvas.draw()

    def on_motion(self, event):
        "Move the ROI if the mouse is over it"
        if self.press is None: return
        if event.inaxes != self.circ.axes: return
        x0, y0, xpress, ypress = self.press
        dx = event.xdata - xpress
        dy = event.ydata - ypress
        self.circ.center = (x0+dx, y0+dy)
        self.circ.figure.canvas.draw()

    def on_release(self, event):
        "On release reset the press data"
        self.press = None
        self.circ.figure.canvas.draw()

    def get_color(self):
        return self.circ.get_facecolor()

class ImgPointSelect(ImageSequence):
    verbose = True
    connected = False
    cw = color_walker()
    def onclick(self,event):
        tb = get_current_fig_manager().toolbar
        if event.inaxes != self.ax1 or tb.mode != '': return

        x,y = round(event.xdata), round(event.ydata)
        if event.button is 1 and \
           not self.any_roi_contains(event):
            
            label = unique_tag(self.roi_labels())
            c = Circle((x,y), 5, alpha = 0.5,
                       label = label,
                       color=self.cw.next())
            c.figure = self.fig
            drc = DraggableCircle(c, self)
            #drc.connect()
            self.drcs[label]=drc
            
            self.ax1.add_patch(c)
        legend()
        draw()

    def any_roi_contains(self,event):
        "Checks if event is contained by any ROI"
        if len(self.drcs) < 1 : return False
        return reduce(lambda x,y: x or y,
                      [roi.circ.contains(event)[0]
                       for roi in self.drcs.values()])
    
    def roi_labels(self):
        "List of labels for all ROIs"
        return self.drcs.keys()
    
    def save_rois(self, fname):
        "Saves picked ROIs to a file"
        pickle.dump(map(struct_circle, [x.circ for x in self.drcs.values()]),
                    file(fname, 'w'))

    def load_rois(self, fname):
        "Load stored ROIs from a file"
        circles = map(circle_from_struct, pickle.load(file(fname)))
        map(self.ax1.add_patch, circles) # add points to the axes
        self.drcs = dict([(c.get_label(), DraggableCircle(c,self))
                          for c in circles])
        legend()
        draw() # redraw the axes
        return

    def pick_rois(self, ch=None, points = []):
        "Start picking up ROIs"
        self.drcs = {}
        self.fig = figure()
        self.ax1 = axes()
        if ch is None: ch=self.ch
        title("Channel: %s" % ('red', 'green')[ch] )
        self.pl = self.ax1.imshow(self.mean(ch),
                                  aspect='equal',
                                  origin='low',
                                  cmap=matplotlib.cm.gray)
        if True or self.connected is False:
            self.fig.canvas.mpl_connect('button_press_event',
                                        self.onclick)
            #self.fig.canvas.mpl_connect('pick_event', self.onpick)
            self.connected = True

    def roi_timeview(self, roi, ch=None, normp=False):
        "Returns timeseries from a single ROI"
        fn = in_circle(roi.center, roi.radius)
        X,Y = meshgrid(*map(range, self.shape[::-1]))
        v = array([mean(frame[fn(X,Y)])
                   for frame in self.ch_view(ch)])
        if normp:
            Lnorm = type(normp) is int and normp or len(v)
            #return (v-np.mean(v))/np.std(v[:Lnorm])
            return (v-np.mean(v[:Lnorm]))/np.std(v[:Lnorm])
        else: return v
        
    def list_roi_timeseries_from_labels(self, roi_labels, **keywords):
        "Returns timeseres for a list of roi labels"
        return [self.roi_timeseries_from_label(label, **keywords)
                for label in roi_labels]
    
    def roi_timeseries_from_label(self, tag, **keywords):
        return self.roi_timeview(self.drcs[tag].circ, **keywords)
        
    def get_timeseries(self, rois=None, ch=None, normp=False):
        rois = ifnot(rois, self.drcs.keys())
        return [self.roi_timeview(self.drcs[p].circ, ch, normp)
                for p in  rois]

    def save_time_series_to_file(self, fname, ch=None, normp = False):
        ch = ifnot(ch, self.ch)
        rois = self.drcs.keys()
        t = self.get_time()
        ts = self.get_timeseries(ch=ch, normp=normp)
        fd = file(fname, 'w')
        out_string = "Channel %d\n" % ch
        out_string += "Time\t" + '\t'.join(rois) + '\n'
        for k in xrange(self.Nf):
            out_string += "%e\t"%t[k]
            out_string += '\t'.join(["%e"%a[k] for a in ts])
            out_string += '\n'
        fd.write(out_string)
        fd.close()


    def show_timeseries(self, rois = None, **keywords):
        t = self.get_time()
        for x,tag,roi,ax in self.roi_show_iterator(rois, **keywords):
            ax.plot(t, x, color = roi.get_facecolor())
            xlim((0,t[-1]))

    def show_ffts(self, rois = None, **keywords):
        L = self.Nf
        freqs = fftfreq(L, self.dt)[1:L/2]
        for x,tag,roi,ax in self.roi_show_iterator(rois, **keywords):
            y = abs(fft(x))[1:L/2]
            ax.plot(freqs, y**2)

    def show_spectrograms(self, rois = None, freqs = None,
                          wavelet = pycwt.Morlet(),
                          vmin = None,
                          vmax = None,
                          normp= True,
                          **keywords):
        keywords.update({'rois':rois, 'normp':normp})
        fs = 1.0/self.dt
        freqs = ifnot(freqs, self.default_freqs())
        axlist = []
        for x,tag,roi,ax in self.roi_show_iterator(**keywords):
            aux.wavelet_specgram(x, freqs, fs, ax,
                                wavelet, vmin=vmin, vmax=vmax)
            axlist.append(ax)
        return axlist

    def setup_axes1(self, figsize = (12,6)):
        "Set up axes for a plot with signal, spectrogram and a colorbar"
        fig = figure(figsize = figsize)
        ax = [fig.add_axes((0.08, 0.4, 0.8, 0.5))]
        ax.append(fig.add_axes((0.08, 0.04, 0.8, 0.3), sharex=ax[0]))
        ax.append(fig.add_axes((0.9, 0.4, 0.02, 0.5), 
                               xticklabels=[], 
                               yticklabels=[]))
        return fig,ax

    
    def show_spectrogram_with_ts(self,
                                 roilabel,
                                 freqs=None,
                                 wavelet = pycwt.Morlet(),
                                 title_string = None,
                                 vmin = None,
                                 vmax = None,
                                 normp = True,
                                 **keywords):
        "Create a figure of a signal, spectrogram and a colorbar"
        signal = self.get_timeseries([roilabel],normp=normp)[0]
        Ns = len(signal)
        f_s = 1/self.dt
        if freqs is None: freqs = self.default_freqs()
        title_string = ifnot(title_string, roilabel)
        tvec = np.arange(0, Ns*self.dt, self.dt)
        L = min(Ns,len(tvec))
        tvec,signal = tvec[:L],signal[:L]
        lc = self.drcs[roilabel].get_color()
        fig,axlist = self.setup_axes1()
        axlist[1].plot(tvec, signal,'-',color=lc)
        axlist[1].set_xlabel('time, s')
        aux.wavelet_specgram(signal, freqs, f_s, axlist[0], vmax=vmax,
                             cax = axlist[2])
        axlist[0].set_title(title_string)
        axlist[0].set_ylabel('Frequency, Hz')
        return fig

    def show_wmps(self, rois = None, freqs = None,
                  wavelet = pycwt.Morlet(),
                  vmin = None,
                  vmax = None,
                  **keywords):
        "show mean wavelet power spectra"
        keywords.update({'rois':rois, 'normp':True})
        fs = 1.0/self.dt
        freqs = ifnot(freqs, self.default_freqs())
        for x,tag,roi,ax in self.roi_show_iterator(**keywords):
            cwt = pycwt.cwt_f(x, freqs, 1./self.dt, wavelet, 'zpd')
            eds = pycwt.eds(cwt, wavelet.f0)
            ax.plot(freqs, np.mean(eds, 1))

    def default_freqs(self, nfreqs = 1024):
        return linspace(4.0/(self.Nf*self.dt),
                      0.5/self.dt, num=nfreqs)

    def roi_show_iterator(self, rois = None,
                              ch = None, normp=False):
            ch = ifnot(ch, self.ch)
            rois = ifnot(rois, self.drcs.keys())
            L = len(rois)
            if L < 1: return
            fig = figure(figsize=(8,4.5), dpi=80)
            for i, roi_label in enumerate(rois):
                roi = self.drcs[roi_label].circ
                ax = fig.add_subplot(L,1,i+1)
                x = self.roi_timeview(roi, ch, normp=normp)
                if i == L-1:
                    ax.set_xlabel("time, sec")
                ax.set_ylabel(roi_label)
                yield x, roi_label, roi, ax
            fig.show()

    def wfnmap(self,extent, nfreqs = 16,
               wavelet = pycwt.Morlet(),
               func = np.mean,
               normL = None,
               ch=None,
               alias=None):
        """
        Wavelet-based 'functional' map of the frame sequence

        Arguments
        ----------

        *extent* is the window of the form
        (start-time, stop-time, low-frequency, high-frequency)

        *nfreqs* -- how many different frequencies in the
        given range (default 16)

        *wavelet* -- wavelet object (default pycwt.Morlet())

        *func* -- function to apply to the wavelet spectrogram within the window
        of interest. Default, np.mean

        *normL* -- length of normalizing part (baseline) of the time series

        *ch* -- color channel / default, reads the self.ch

        *alias* -- if 0, no alias, then each frame is filtered, if >0,
        average +- pixels, if an array, use this as akernel to convolve each
        frame with; see aliased_pix_iter2 for default kernel
        """
        tick = time.clock()
        out = ones(self.shape, np.float64)
        total = self.shape[0]*self.shape[1]
        k = 0
        freqs = linspace(*extent[2:], num=nfreqs)
        pix_iter = None
        normL = ifnot(normL, self.Nf)

        if type(alias) == np.ndarray or alias is None:
            pix_iter = self.aliased_pix_iter2(ch,alias)
        elif alias == 0:
            pix_iter = self.simple_pix_iter(ch)
        elif alias >0:
            pix_iter = self.aliased_pix_iter(ch,alias)
        
        start,stop = [int(a/self.dt) for a in extent[:2]]
        for s,i,j in pix_iter:
            s = s-mean(s[:normL])
            cwt = pycwt.cwt_f(s, freqs, 1./self.dt, wavelet, 'zpd')
            eds = pycwt.eds(cwt, wavelet.f0)/std(s[:normL])**2
            x=func(eds[:,start:stop])
            #print "\n", start,stop,eds.shape, "\n"
            out[i,j] = x
            k+= 1
            if self.verbose:
                sys.stderr.write("\rpixel %05d of %05d"%(k,total))
        if self.verbose:
            sys.stderr.write("\n Finished in %3.2f s\n"%(time.clock()-tick))
        return out

    def setup_freqs(self,freqs):
        if freqs is None:
            if hasattr(self,'freqs'):
                freqs = self.freqs
            else:
                freqs = self.default_freqs()
                self.freqs = freqs
        return freqs

    def show_xwt_roi(self, roi1, roi2, freqs=None, ch=None,
                     func = pycwt.wtc_f,
                     wavelet = pycwt.Morlet()):
        "show cross wavelet spectrum or wavelet coherence for two ROI"
        freqs = self.setup_freqs(freqs)
        self.extent=[0,self.Nf*self.dt, freqs[0], freqs[-1]]
        self.time = self.get_time()

        s1 = self.roi_timeview(roi1,ch,True)
        s2 = self.roi_timeview(roi2,ch,True)
        res = func(s1,s2, freqs,1.0/self.dt,wavelet)

        t = self.get_time()

        figure();
        ax1= subplot(211);
        plot(t,s1,color=roi1.get_facecolor(),
             label=roi1.get_label())
        plot(t,s2,color=roi2.get_facecolor(),
             label = roi2.get_label())
        legend()
        ax2 = subplot(212, sharex = ax1);
        ext = (t[0], t[-1], freqs[0], freqs[-1])
        ax2.imshow(res, extent = ext, cmap = aux.swanrgb())
        #self.cone_infl(freqs,wavelet)
        #self.confidence_contour(res,2.0)

    def ffnmap(self, fspan, ch=None, alias = None, func=np.mean):
        tick = time.clock()
        out = ones(self.shape, np.float64)
        total = self.shape[0]*self.shape[1]
        k = 0
        freqs = fftfreq(self.Nf, self.dt)
        pix_iter = self.aliased_pix_iter2(ch, alias)
        fstart,fstop = fspan
        fmask = (freqs >= fstart)*(freqs < fstop)
        for s,i,j in pix_iter:
            s = s-mean(s)
            s_hat = fft(s)
            x = (abs(s_hat)/std(s))**2
            out[i,j] = func(x[fmask])
            k+=1
            if self.verbose:
                sys.stderr.write("\rpixel %05d of %05d"%(k,total))
        if self.verbose:
            sys.stderr.write("\n Finished in %3.2f s\n"%(time.clock()-tick))
        return out

    def show_xwt(self, **kwargs):
        for p in aux.allpairs0(self.drcs.values()):
            self.show_xwt_roi(p[0].circ,p[1].circ,**kwargs)
           
            
            
                
                    

            
        

        
