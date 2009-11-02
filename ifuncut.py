#!/usr/bin/python

import os
import numpy
import glob
from itertools import combinations
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

def threshold(mat, thresh, func=lambda a,b: a>b):
    return func(mat, thresh)

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


class ImgPointSelect(ImageSequence):
    verbose = True
    connected = False
    cw = color_walker()
    def onclick(self,event):
        tb = get_current_fig_manager().toolbar
        if event.inaxes == self.ax1 and tb.mode=='':
            x,y = round(event.xdata), round(event.ydata)
            if event.button in [1]:
                self.points.append(Circle((x,y), 5, alpha = 0.5,
                                          color=self.cw.next()))
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
        if ch is None: ch=self.ch
        title("Channel: %s" % ('red', 'green')[ch] )
        self.pl = self.ax1.imshow(self.mean(ch),
                                  aspect='equal',
                                  cmap=matplotlib.cm.gray)
        if True or self.connected is False:
            self.fig.canvas.mpl_connect('button_press_event',
                                        self.onclick)
            self.fig.canvas.mpl_connect('scroll_event',
                                        self.onscroll)
            self.connected = True

    def make_timeview(self, point, ch=None, normp=False):
        fn = in_circle(point.center, point.radius)
        X,Y = meshgrid(*map(range, self.shape[::-1]))
        v = array([mean(frame[fn(X,Y)]) for frame in \
                   self.ch_view(ch)])
        if normp: return v/numpy.std(v)
        else: return v
        
    def get_timeseries(self, ch=None, normp=False):
        return (self.make_timeview(p, ch, normp) for p in self.points)

    def show_timeseries(self, ch=None):
        figure()
        t = arange(0,self.Nf*self.dt, self.dt)[:self.Nf]
        L = len(self.points)
        if ch is None: ch=self.ch
        for i in xrange(L):
            subplot(L,1,i+1);
            if i == 0:
                title("Channel: %s" % ('red', 'green')[ch])
            p = self.points[i]
            x = self.make_timeview(p, ch)
            plot(t, x, color=p.get_facecolor())
            xlim((0,t[-1]))

    def make_cwt(self, freqs,
                 ch = None,
                 wavelet=pycwt.Morlet()):
        self.wcoefs =  [pycwt.cwt_f(v, freqs, 1.0/self.dt,
                                    wavelet)\
                        for v in self.get_timeseries(ch,normp=True)]
        return self.wcoefs

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
               vmin = vmin, vmax=vmax,
               cmap = matplotlib.cm.jet, alpha=0.95, hold=True)
    def confidence_contour(self, esurf, L=3.0):
        # Show 95% confidence level (against white noise, v=3 \sigma^2)
        contour(esurf, [3.0], extent=self.extent,
                cmap=matplotlib.cm.gray)

    def default_freqs(self):
        return arange(3.0/(self.Nf*self.dt),
                      0.5/self.dt, 0.005)
    def show_spectrograms(self,
                          freqs = None,#arange(0.1, 5, 0.005),
                          ch = None,
                          wavelet = pycwt.Morlet(),
                          vmin = None,
                          vmax = None):
        figure()
        if freqs is None: freqs = self.default_freqs()
        self.freqs = freqs
        self.time = arange(0,self.Nf*self.dt, self.dt)[:self.Nf]
        L = len(self.points)
        self.extent=[0,self.Nf*self.dt, freqs[0], freqs[-1]]
        self.make_cwt(freqs, ch=ch, wavelet=wavelet)

        if ch is None: ch=self.ch

        for i in xrange(L):
            subplot(L,1,i+1);
            if i == 0:
                title("Channel: %s" % ('red', 'green')[ch] )
                    
            esurf = pycwt.eds(self.wcoefs[i],wavelet.f0)
            self.specgram(esurf,vmin,vmax)
            colorbar()
            self.cone_infl(freqs, wavelet)
            self.confidence_contour(esurf)

            
        xlabel('time, s'); ylabel('Freq., Hz')

    def setup_freqs(self,freqs):
        if freqs is None:
            if hasattr(self,'freqs'):
                freqs = self.freqs
            else:
                freqs = self.default_freqs()
                self.freqs = freqs
        return freqs

    def show_xwt_roi(self, p1,p2,freqs=None, ch=None,
                     func = pycwt.wtc_f,
                     wavelet = pycwt.Morlet()):
        "show cross wavelet spectrum or wavelet coherence for two ROI"
        freqs = self.setup_freqs(freqs)
        self.extent=[0,self.Nf*self.dt, freqs[0], freqs[-1]]
        self.time = arange(0,self.Nf*self.dt, self.dt) #codeduplicate

        s1 = self.make_timeview(p1,ch,True)
        s2 = self.make_timeview(p2,ch,True)
        res = func(s1,s2, freqs,1.0/self.dt,wavelet)

        figure();
        ax1= subplot(211);
        plot(self.time,s1,color=p1.get_facecolor())
        plot(self.time,s2,color=p2.get_facecolor())
        subplot(212, sharex = ax1);
        self.specgram(res)
        self.cone_infl(freqs,wavelet)
        self.confidence_contour(res,2.0)

    def show_xwt(self, **kwargs):
        for p in allpairs0(self.points):
            self.show_xwt_roi(p[0],p[1],**kwargs)
           
            
            
                
                    

            
        

        
