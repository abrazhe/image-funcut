#!/usr/bin/python

import os
import numpy
import glob
from itertools import combinations
from pylab import *
from pylab import mpl
from swan import pycwt

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
    minv = numpy.min(mat)
    maxv = numpy.max(mat)
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
    x = numpy.std(mat)
    return func(mat, x*n)


def image_names(pat):
    x = glob.glob(pat)
    x.sort()
    return x

def cone_infl(freqs, wavelet, endtime):
    "Draw shaded regions on edges of a spectrogram to mark edge errors"
    try:
        fill_betweenx(freqs,0,wavelet.cone(freqs),
                      alpha=0.5, color='black')
        fill_betweenx(freqs,
                      endtime+wavelet.cone(-freqs),
                      endtime,
                      alpha=0.5, color='black')
    except:
        print("Can't use fill_betweenx function: update\
        maptlotlib?")


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
        #self.d = array([imread(f) for f in image_names(pattern)])
        self.d = [imread(f) for f in image_names(pattern)]
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
        return numpy.mean(self.ch_view(ch), axis=0)

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

    def disconnect(self):
        "Disconnect stored connections"
        map(self.circ.figure.canvas.mpl_disconnect,
            (self.cidpress, self.cidrelease, self.cidscroll,
             self.cidmotion))

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
        if normp: return v/numpy.std(v)
        else: return v
        
    def get_all_timeseries(self, ch=None, normp=False):
        return (self.roi_timeview(p.circ, ch, normp)
                for p in  self.drcs.values())


    def show_timeseries(self, rois=None, ch=None, normp=False):
        t = self.get_time()
        if ch is None: ch=self.ch
        if rois == None:
            rois = self.drcs.keys()

        L = len(rois)
        if L < 1 : return

        fig = figure(figsize=(8,4.5), dpi=80)
        for i, roi_label in enumerate(rois):
            roi = self.drcs[roi_label].circ
            ax = fig.add_subplot(L,1,i+1)
            ax.plot(t,
                    self.roi_timeview(roi),
                    color = roi.get_facecolor())
            ax.set_ylabel(roi.get_label())
            xlim((0,t[-1]))
            if i == L-1:
                ax.set_xlabel("time, sec")
        fig.show()

    def roi_cwt(self, roi,
                 freqs,
                 ch = None,
                 wavelet=pycwt.Morlet()):
        return pycwt.cwt_f(self.roi_timeview(roi,ch,normp=True),
                           freqs, 1.0/self.dt, wavelet, 'cpd')

    def cone_infl(self,freqs, wavelet):
        try:
            fill_betweenx(freqs,0,wavelet.cone(freqs),
                          alpha=0.5, color='black')
            fill_betweenx(freqs,
                          self.time[-1]+wavelet.cone(-freqs),
                          self.time[-1],
                          alpha=0.5, color='black')
        except:
            print("Can't use fill_betweenx function: update\
            maptlotlib?")

    def specgram(self,esurf,vmin=None,vmax=None):
        imshow(esurf, extent=self.extent,
               origin='low',
               vmin = vmin, vmax=vmax,
               cmap = matplotlib.cm.jet, alpha=0.95, hold=True)
    def confidence_contour(self, esurf, L=3.0):
        # Show 95% confidence level (against white noise, v=3 \sigma^2)
        contour(esurf, [L], extent=self.extent,
                cmap=matplotlib.cm.gray)

    def default_freqs(self):
        return arange(3.0/(self.Nf*self.dt),
                      0.5/self.dt, 0.005)

    def show_spectrograms(self,
                          rois = None,
                          freqs = None,#arange(0.1, 5, 0.005),
                          ch = None,
                          wavelet = pycwt.Morlet(),
                          vmin = None,
                          vmax = None):
        fig = figure()
        if rois is None: rois = self.drcs.keys()
        if freqs is None: freqs = self.default_freqs()
        self.freqs = freqs
        self.time = self.get_time()
        self.extent=[0,self.Nf*self.dt, freqs[0], freqs[-1]]
        #self.make_all_cwt(freqs, ch=ch, wavelet=wavelet)

        if ch is None: ch=self.ch

        L = len(rois)
        if L < 1: return
        
        for i,roi_label in enumerate(rois):
            roi = self.drcs[roi_label].circ
            ax = fig.add_subplot(L,1,i+1);
            if i == 0:
                ax.set_title("Channel: %s" % ('red', 'green')[ch] )
            elif i == L-1:
                ax.set_xlabel("time, sec")

            wcoefs = self.roi_cwt(roi, freqs, ch, wavelet) 
            esurf = pycwt.eds(wcoefs, wavelet.f0)
            self.specgram(esurf,vmin,vmax)
            colorbar()
            self.cone_infl(freqs, wavelet)
            self.confidence_contour(esurf)
            ax.set_ylabel(roi_label)

            
        #xlabel('time, s'); ylabel('Freq., Hz')

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
        subplot(212, sharex = ax1);
        self.specgram(res)
        self.cone_infl(freqs,wavelet)
        self.confidence_contour(res,2.0)

    def show_xwt(self, **kwargs):
        for p in allpairs0(self.drcs.values()):
            self.show_xwt_roi(p[0].circ,p[1].circ,**kwargs)
           
            
            
                
                    

            
        

        
