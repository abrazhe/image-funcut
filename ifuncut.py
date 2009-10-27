#!/usr/bin/python

import os
import numpy
import glob
from pylab import *
from iwavelets import pycwt
from swan_gui import swan

def best (scoref, lst):
    n,winner = 0, lst[0]
    for i, item in enumerate(lst):
        if  scoref(item, winner): n, winner = i, item
    return n,winner

def min1(scoref, lst):
    return best(lambda x,y: x < y, map(scoref, lst))

def nearest_item_ind(items, xy, fn = lambda a: a):
    "Index of nearest item from collection. collection, position, selector"
    return min1(lambda p: eu_dist(fn(p), xy), items)



def eu_dist(p1,p2):
    return sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def line_from_points(p1,p2):
    p1 = map(float, p1)
    p2 = map(float, p2)
    k = (p1[1] - p2[1])/(p1[0] - p2[0])
    b = p1[1] - p1[0]*k
    return lambda x: k*x + b

def in_circle(xy, radius):
    return lambda x,y: eu_dist((x,y), xy) < radius

def in_ellipse1(f1,f2):
    return lambda x,y: \
           (eu_dist((x,y), f1) + eu_dist((x,y), f2)) < eu_dist(f1,f2)

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

def times_std(mat, n, func=lambda a,b: a>b):
    x = numpy.std(mat)
    return func(mat, x*n)


def image_names(pat):
    x = glob.glob(pat)
    x.sort()
    return x

class ImageSequence:
    def __init__(self, pat, channel=0, dt = 0.15):
        self.pat = pat
        self.ch = channel
        self.dt = dt
        self.d = [imread(f) for f in image_names(pat)]
        self.Nf = len(self.d)
        self.shape = self.d[0].shape[:-1]

    def ch_view(self, ch=None):
        if ch is None: ch=self.ch
        return (d[:,:,ch] for d in self.d)

    def mean(self, ch=0):
        return numpy.mean(list(self.ch_view(ch)), axis=0)

    def show(self, ch=0):
        self.fig = figure()
        self.ax = axes()
        self.pl = imshow(self.mean(ch), aspect='equal', cmap=matplotlib.cm.gray)

def rezip(a):
    return zip(*a)

def shorten_movie(m,n):
    return array([mean(m[i:i+3,:],0) for i in xrange(0,len(m), n)])


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
            extent = (0, int(eu_dist(*self.endpoints)),
                      0, self.Nf*self.dt),
            interpolation='nearest')
        
    def get_timeview(self):
        return self.make_timeview()


class ImgPointSelect(ImageSequence):
    verbose = True
    connected = False
    def onclick(self,event):
        tb = get_current_fig_manager().toolbar
        if event.inaxes == self.ax1 and tb.mode=='':
            x,y = round(event.xdata), round(event.ydata)
            if event.button in [1]:
                self.points.append(Circle((x,y), 5, alpha = 0.5,
                                          color=[rand(),rand(),rand()]))
                self.ax1.add_patch(self.points[-1])
            else:
                n,p = nearest_item_ind(self.points, (x,y), lambda p: p.center)
                p = self.points.pop(n)
                p.remove()
            draw()
    def onscroll(self, event):
        if event.inaxes == self.ax1:
            step = 1
            x,y = round(event.xdata), round(event.ydata)
            n,p = nearest_item_ind(self.points, (x,y), lambda p: p.center)
            p = self.points[n]
            r = p.get_radius()
            if event.button in ['up']:
                p.set_radius(r+step)
            else:
                p.set_radius(max(0.1,r-step))
            draw()
            
    def select(self, ch):
        self.points = []
        self.fig = figure()
        self.ax1 = axes()
        self.pl = self.ax1.imshow(self.mean(ch),
                                  aspect='equal',
                                  cmap=matplotlib.cm.gray)
        if True or self.connected is False:
            self.fig.canvas.mpl_connect('button_press_event',
                                        self.onclick)
            self.fig.canvas.mpl_connect('scroll_event',
                                        self.onscroll)
            self.connected = True

    def make_timeview(self, point, ch=None):
        fn = in_circle(point.center, point.radius)
        X,Y = meshgrid(*map(range, (self.shape)))
        return array([mean(frame[fn(X,Y)]) for frame in self.ch_view(ch)])
    
    def get_timeseries(self, ch=None):
        return (self.make_timeview(p, ch) for p in self.points)

    def show_timeseries(self, ch=None):
        figure()
        t = arange(0,self.Nf*self.dt, self.dt)
        L = len(self.points)
        print ch
        for i in xrange(L):
            subplot(L,1,i+1);
            p = self.points[i]
            x = self.make_timeview(p, ch)
            plot(t, x, color=p.get_facecolor())

    def make_cwt(self, freqs,
                 ch = None,
                 wavelet=pycwt.Morlet()):
        self.wcoefs =  [pycwt.cwt_f(v, freqs, 1.0/self.dt, wavelet) for v
                     in self.get_timeseries(ch)]
        return self.wcoefs

    def show_spectrograms(self,
                          freqs = None,#arange(0.1, 5, 0.005),
                          ch = None):
        figure()
        if freqs is None:
            freqs = arange(2.0/(self.Nf*self.dt),
                           0.5/self.dt, 0.005)
        t = arange(0,self.Nf*self.dt, self.dt)
        L = len(self.points)
        extent=[0,self.Nf*self.dt, freqs[0], freqs[-1]]
        self.make_cwt(freqs, ch=ch)
        for i in xrange(L):
            subplot(L,1,i+1);
            imshow(EDS(self.wcoefs[i]), extent=extent,
                   cmap = matplotlib.cm.jet)
        
            
            
                
                    
def EDS(x, f0=1.5):
    return (x.real**2 + x.imag**2)/f0

def crossw(x,y):
    return x*y.conjugate()

def coherence(x,y,f0=1.5):
    return abs(crossw(x,y))**2/(abs(x)*abs(y))

def cphase(x,y):
    d = crossw(x,y)
    return arctan2(d.imag, d.real)

def cwt_phase(x):
    return arctan2(x.imag, x.real)

            
        

        
