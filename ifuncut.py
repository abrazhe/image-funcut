#!/usr/bin/python

import os
import numpy as np
import glob
from itertools import combinations
from pylab import *
from pylab import mpl
from swan import pycwt

from imfun import aux_utils as aux

from imfun.aux_utils import ifnot

mpl.rcParams['image.aspect'] = 'auto'
mpl.rcParams['image.origin'] = 'lower'

def best (scoref, lst):
    if len(lst) > 0:
        n,winner = 0, lst[0]
        for i, item in enumerate(lst):
            if  scoref(item, winner): n, winner = i, item
            return n,winner
    else: return -1,None

def min1(scoref, lst):
    return best(lambda x,y: x < y, map(scoref, lst))

def nearest_item_ind(items, xy, fn = lambda a: a):
    "Index of nearest item from collection. collection, position, selector"
    return min1(lambda p: eu_dist(fn(p), xy), items)

def allpairs0(seq):
    return list(combinations(seq,2))

def allpairs(seq):
    if len(seq) <= 1: return []
    else:
        return [[seq[0], s] for s in seq[1:]] + allpairs(seq[1:])


def eu_dist(p1,p2):
    return sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def line_from_points(p1,p2):
    p1 = map(float, p1)
    p2 = map(float, p2)
    k = (p1[1] - p2[1])/(p1[0] - p2[0])
    b = p1[1] - p1[0]*k
    return lambda x: k*x + b

def in_circle(coords, radius):
    return lambda x,y: eu_dist((x,y), coords) < radius

#def in_ellipse1(f1,f2):
#    return lambda x,y: \
#           (eu_dist((x,y), f1) + eu_dist((x,y), f2)) < eu_dist(f1,f2)

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
    def __init__(self, pattern, channel=0, dt = 0.16):
        """
        Creates an instance. Image files must be in RGB format
        - pattern: a string like "/path/to/files/*.tif"
        - channel: an integer, 0--2, sets default channel. 0 is red, 1 is green
        and 2 is blue.
        - dt: a float, specifies time delay between the frames
        """
        self.ch = channel
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
        if ch is None: ch=self.ch
#        return self.d[:,:,:,ch]
        return array([d[:,:,ch] for d in self.d])

    def get_time(self):
        """
        Return a vector of time values, calculated from sampling interval and
        number of frames
        """
        return arange(0,self.Nf*self.dt, self.dt)[:self.Nf]


    def mean(self, ch=0):
        """
        Returns 'mean frame' along the sequence in a given color channel as a
        2D numpy array
        """
        return np.mean(self.ch_view(ch), axis=0)

    def show(self, ch=0):
        """
        Shows a mean frame along the sequence
        """
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
        if normp: return (v-np.mean(v))/np.std(v)
        else: return v
        
    def list_rois_timeseries_from_labels(self, roi_labels, **keywords):
        "Returns timeseres for a list of roi labels"
        return [self.roi_timeseries_from_label(label, **keywords)
                for label in roi_labels]
    
    def roi_timeseries_from_label(self, roilabel, **keywords):
        return self.roi_timeview(self.drcs[roilabel].circ, **keywords)
        
    def get_timeseries(self, rois=None, ch=None, normp=False):
        if not rois:
            rois = self.drcs.keys()
        return [self.roi_timeview(p.circ, ch, normp)
                for p in  rois]


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
                          **keywords):
        keywords.update({'rois':rois, 'normp':True})
        fs = 1.0/self.dt
        freqs = ifnot(freqs, self.default_freqs())
        for x,tag,roi,ax in self.roi_show_iterator(**keywords):
            aux.wavelet_specgram(x, freqs, fs, ax,
                                wavelet, vmin=vmin, vmax=vmax)

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



    def roi_cwt(self, roi,
                 freqs,
                 ch = None,
                 wavelet=pycwt.Morlet()):
        return pycwt.cwt_f(self.roi_timeview(roi,ch,normp=True),
                           freqs, 1.0/self.dt, wavelet, 'cpd')


    def default_freqs(self):
        return arange(3.0/(self.Nf*self.dt),
                      0.5/self.dt, 0.005)

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

        figure();
        ax1= subplot(211);
        plot(self.time,s1,color=roi1.get_facecolor(),
             label=roi1.get_label())
        plot(self.time,s2,color=roi2.get_facecolor(),
             label = roi2.get_label())
        legend()
        ax2 = subplot(212, sharex = ax1);
        specgram(res,self.extent, ax2)
        self.cone_infl(freqs,wavelet)
        self.confidence_contour(res,2.0)

    def show_xwt(self, **kwargs):
        for p in allpairs0(self.drcs.values()):
            self.show_xwt_roi(p[0].circ,p[1].circ,**kwargs)
           
            
            
                
                    

            
        

        
