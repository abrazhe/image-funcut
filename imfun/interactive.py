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


import itertools

def ar1(alpha = 0.74):
    "Simple auto-regression model"
    prev = randn()
    while True:
        res = prev*alpha + randn()
        prev = res
        yield res


def circle_from_struct(circ_props):
    cp = circ_props.copy()
    _  = cp.pop('func')
    center = cp.pop('center')
    return Circle(center, **cp)

def line_from_struct(line_props):
    lp = line_props.copy()
    _ = lp.pop('func')
    xdata, ydata = lp.pop('xdata'), lp.pop('ydata')
    return Line2D(xdata, ydata,**lp)


import pickle
import random

vowels = "aeiouy"
consonants = "qwrtpsdfghjklzxcvbnm"


def rand_tag():
    return ''.join(map(random.choice,
                       (consonants, vowels, consonants)))

def unique_tag(tags, max_tries = 1e4):
    n = 0
    while n < max_tries:
        tag = rand_tag()
        n += 1
        if not tag in tags:
            return tag
    return "Err"


def nearest_item_ind(items, xy, fn = lambda a: a):
    """
    Index of the nearest item from a collection.
    Arguments: collection, position, selector
    """
    return aux.min1(lambda p: eu_dist(fn(p), xy), items)



def eu_dist(p1,p2):
    "Euler distance between two points"
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
    return array([data[f(p1[1],p2[1])(i), f(p1[0],p2[0])(i)]
                  for i in range(L)])


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

def color_walker():
    red, green, blue = ar1(), ar1(), ar1()
    while True:
        yield map(lambda x: mod(x.next(),1.0), (red,green,blue))



def rezip(a):
    return zip(*a)

def shorten_movie(m,n):
    return array([mean(m[i:i+n,:],0) for i in xrange(0, len(m), n)])

def view_fseq_frames(fseq):
    f = figure()
    axf = axes()
    frame_index = [0]
    frames = fseq.aslist()
    Nf = len(frames)
    plf = axf.imshow(frames[0],
                     aspect = 'equal', cmap=mpl.cm.gray)
    def skip(event,n=1):
        fi = frame_index[0]
        key = hasattr(event, 'button') and event.button or event.key
        if key in (4,'4','down'):
            fi -= n
        elif key in (5,'5','up'):
            fi += n
        fi = fi%Nf
        plf.set_data(frames[fi])
        axf.set_xlabel('%03d'%fi)
        frame_index[0] = fi
        f.canvas.draw()
    f.canvas.mpl_connect('scroll_event',skip)
    f.canvas.mpl_connect('key_press_event',lambda e: skip(e,10))


class DraggableObj:
    verbose = True
    def __init__(self, obj, parent,):
        self.obj = obj
        self.parent = parent
        self.connect()
        self.pressed = None
        self.tag = obj.get_label() # obj must support this
        pass

    def redraw(self):
        self.obj.axes.figure.canvas.draw()

    def event_ok(self, event, should_contain=False):
        containsp = True
        if should_contain:
            containsp, _ = self.obj.contains(event)
        return event.inaxes == self.obj.axes and \
               containsp and \
               get_current_fig_manager().toolbar.mode ==''

    def connect(self):
        "connect all the needed events"
        cf = self.obj.axes.figure.canvas.mpl_connect
        self.cid = {
            'press': cf('button_press_event', self.on_press),
            'release': cf('button_release_event', self.on_release),
            'motion': cf('motion_notify_event', self.on_motion),
            'scroll': cf('scroll_event', self.on_scroll),
            'type': cf('key_press_event', self.on_type)
            }
    def on_motion(self, event):
        if not (self.event_ok(event, False) and self.pressed):
            return
        p = event.xdata,event.ydata
        self.move(p)
        self.redraw()

    def on_release(self, event):
        if not self.event_ok(event):
            return
        self.pressed = None
        self.redraw()

    def on_scroll(self, event): pass

    def on_type(self, event): pass
    
    def on_press(self, event): pass

    def disconnect(self):
        map(self.obj.axes.figure.canvas.mpl_disconnect,
            self.cid.values())
    def get_color(self):
        return self.obj.get_facecolor()


class LineScan(DraggableObj):
    def endpoints(self):
        return rezip(self.obj.get_data())
    def length(self):
        return eu_dist(*self.endpoints())
    def check_endpoints(self):
        "Return endpoints in the correct order"
        pts = self.endpoints()
        if pts[0][0] > pts[1][0]:
            return pts[::-1]
        else:
            return pts
    def move(self, p):
        xp,yp   = self.pressed
        dx,dy = p[0] - xp, p[1] - yp
        x0, y0 = self.obj.get_xdata(), self.obj.get_ydata()
        dist1,dist2 = [eu_dist(p,x) for x in self.endpoints()]
        if dist1/(dist1 + dist2) < 0.05:
            dx,dy = array([dx, 0]), array([dy, 0])
        elif dist2/(dist1 + dist2) < 0.05:
            dx,dy = array([0, dx]), array([0, dy])
        self.obj.set_data((x0 + dx,y0 + dy))
        self.pressed = p

    def on_press(self, event):
        if not self.event_ok(event, True):
            return
        if event.button == 3:
            self.parent.roi_objs.pop(self.tag) 
            self.disconnect()
            self.obj.remove()
            self.redraw()
        elif event.button == 1:
            x, y = self.obj.get_xdata(), self.obj.get_ydata()
            x0 = x[1] - x[0]
            y0 = y[1] - y[0]
            self.pressed = event.xdata, event.ydata
        elif event.button == 2:
            self.show_timeview()

    def get_timeview(self):
        points = self.check_endpoints()
        timeview = array([extract_line2(frame, *points) for frame in
                          self.parent.fseq.frames()])
        return timeview,points
    def show_timeview(self):
        timeview,points = self.get_timeview()
        if timeview is not None:
            ax = pl.axes();
            ax.imshow(timeview,
                      extent=(0, timeview.shape[1], 0,
                              self.parent.fseq.dt*self.parent.length()),
                      aspect='equal')
            ax.set_ylabel('time, sec')
            ax.set_title('Timeview for '+ self.tag)
            ax.figure.show()

    def to_struct(self):
        l = self.obj
        return {'func': line_from_struct,
                'xdata': l.get_xdata(),
                'ydata': l.get_ydata(),
                'alpha': l.get_alpha(),
                'label': l.get_label(),
                'color': l.get_color(),}


class CircleROI(DraggableObj):
    "Draggable Circle ROI"
    step = 1
    def on_scroll(self, event):
        if not self.event_ok(event, True): return
        r = self.obj.get_radius()
        if event.button in ['up']:
            self.obj.set_radius(r+self.step)
        else:
            self.obj.set_radius(max(0.1,r-self.step))
        self.redraw()
        return

    def on_type(self, event):
        if not self.event_ok(event, True): return
        if self.verbose:
            print event.key
        tags = [self.tag]
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
        if not self.event_ok(event, True): return
        x0,y0 = self.obj.center
        if event.button is 1:
            self.pressed = event.xdata, event.ydata, x0, y0
        elif event.button is 2:
            self.parent.show_timeseries([self.tag])
            
        elif event.button is 3:
            p = self.parent.roi_objs.pop(self.tag)
            self.obj.remove()
            self.disconnect()
            self.obj.axes.legend()
            self.redraw()

    def move(self, p):
        "Move the ROI when the mouse is pressed over it"
        xp,yp, x, y = self.pressed
        dx = p[0] - xp
        dy = p[1] - yp
        self.obj.center = (x+dx, y+dy)
        self.redraw()

    def get_timeview(self, normp=False):
        roi = self.obj
        fn = in_circle(roi.center, roi.radius)
        shape = self.parent.fseq.shape()
        X,Y = meshgrid(*map(range, shape))
        v = self.parent.fseq.mask_reduce(fn(X,Y))
        if normp:
            Lnorm = type(normp) is int and normp or len(v)
            return (v-np.mean(v[:Lnorm]))/np.std(v[:Lnorm])
        else: return v

    def to_struct(self):
        c = self.obj
        return {'func' : circle_from_struct,
                'center': c.center,
                'radius': c.radius,
                'alpha': c.get_alpha(),
                'label': c.get_label(),
                'color': c.get_facecolor(),}


class Picker:
    verbose = True
    connected = False
    cw = color_walker()

    def __init__(self, fseq):
        self.fseq = fseq
        self.dt = fseq.dt
        self._Nf = None
        self.roi_objs = {}
        self.min_length = 5

        pass

    def length(self):
        if self._Nf is None:
            self._Nf = self.fseq.length()
        return self._Nf

    
    
    def onclick(self,event):
        tb = get_current_fig_manager().toolbar
        if event.inaxes != self.ax1 or tb.mode != '': return

        x,y = round(event.xdata), round(event.ydata)
        if event.button is 1 and \
           not self.any_roi_contains(event):
            
            label = unique_tag(self.roi_tags())
            c = Circle((x,y), 5, alpha = 0.5,
                       label = label,
                       color=self.cw.next())
            c.figure = self.fig
            self.ax1.add_patch(c)
            print c.axes
            self.roi_objs[label]= CircleROI(c, self)
            #drc.connect()
            
        self.ax1.legend()
        draw()

    def on_press(self, event):
        if event.inaxes !=self.ax1 or \
               self.any_roi_contains(event) or \
               event.button != 3 or \
               get_current_fig_manager().toolbar.mode !='':
            return
        self.pressed = event.xdata, event.ydata
        axrange = self.ax1.get_xbound() + self.ax1.get_ybound()
        tag = unique_tag(self.roi_tags())
        self.curr_line_handle, = self.ax1.plot([0],[0],'-',
                                               label = tag,
                                               color=self.cw.next())
        self.ax1.axis(axrange)
        return

    def on_motion(self, event):
        if (self.pressed is None) or (event.inaxes != self.ax1):
            return
        pstop = event.xdata, event.ydata
        self.curr_line_handle.set_data(*rezip((self.pressed,pstop)))
        self.fig.canvas.draw() #todo BLIT!
        
    def on_release(self,event):
        self.pressed = None
        if not event.button == 3: return
        if self.any_roi_contains(event): return
        if not hasattr(self, 'curr_line_handle'): return
        tag = self.curr_line_handle.get_label()
        if len(self.curr_line_handle.get_xdata()) > 1:
            newline = LineScan(self.curr_line_handle, self)
            if newline.length() > self.min_length:
                self.roi_objs[tag] = newline
            else:
                self.curr_line_handle.remove()
        else:
            self.curr_line_handle.remove()
        if len(self.roi_objs) > 0:
            self.ax1.legend()
        self.fig.canvas.draw() #todo BLIT!
        return


    ## def any_line_contains(self,event):
    ##     "Checks if event is contained by any ROI"
    ##     if len(self.roi_objs) < 1 : return False
    ##     return reduce(lambda x,y: x or y,
    ##                   [roi.obj.contains(event)[0]
    ##                    for roi in self.roi_objs.values()])


    def any_roi_contains(self,event):
        "Checks if event is contained by any ROI"
        if len(self.roi_objs) < 1 : return False
        return reduce(lambda x,y: x or y,
                      [roi.obj.contains(event)[0]
                       for roi in self.roi_objs.values()])
    
    def roi_tags(self):
        "List of tags for all ROIs"
        return self.roi_objs.keys()
    
    def save_rois(self, fname):
        "Saves picked ROIs to a file"
        pickle.dump([x.to_struct() for x in self.roi_objs.values()],
                    file(fname, 'w'))
#                     if isinstance(x, CircleROI)],


    def load_rois(self, fname):
        "Load stored ROIs from a file"
        saved_rois = pickle.load(file(fname))
        rois = [x['func'](x) for x in saved_rois]
        circles = filter(lambda x: isinstance(x, Circle), rois)
        lines = filter(lambda x: isinstance(x, Line2D), rois)
        map(self.ax1.add_patch, circles) # add points to the axes
        map(self.ax1.add_line, lines) # add points to the axes
        self.roi_objs.update(dict([(c.get_label(), CircleROI(c,self))
                                   for c in circles]))
        self.roi_objs.update(dict([(l.get_label(), LineScan(l,self))
                                   for l in lines]))

        self.ax1.legend()
        draw() # redraw the axes
        return

    def start(self, roi_objs={}, ax=None):
        "Start picking up ROIs"
        self.drcs = {}
        self.ax1 = ifnot(ax,axes())
        self.fig = self.ax1.figure

        if hasattr(self.fseq, 'ch'):
            title("Channel: %s" % ('red', 'green')[self.fseq.ch] )
        self.pl = self.ax1.imshow(self.fseq.mean_frame(),
                                  interpolation = 'nearest',
                                  aspect='equal',
                                  cmap=matplotlib.cm.gray)
        if True or self.connected is False:
            self.pressed = None
            self.fig.canvas.mpl_connect('button_press_event',
                                        self.onclick)
            self.fig.canvas.mpl_connect('button_press_event',self.on_press)
            self.fig.canvas.mpl_connect('button_release_event',self.on_release)
            self.fig.canvas.mpl_connect('motion_notify_event',self.on_motion)
            self.connected = True

            #self.fig.canvas.mpl_connect('pick_event', self.onpick)
            self.connected = True
        return self.ax1, self.pl

        
    def list_roi_timeseries_from_labels(self, roi_tags, **keywords):
        "Returns timeseres for a list of roi labels"
        return [self.roi_timeseries_from_label(tag, **keywords)
                for tag in roi_tags]

    def isCircleROI(self,tag):
        return isinstance(self.roi_objs[tag], CircleROI)
    
    def get_timeseries(self, rois=None, normp=False):
        rois = ifnot(rois,
                     filter(self.isCircleROI, self.roi_objs.keys()))
        return [self.roi_objs[tag].get_timeview(normp)
                for tag in  rois]

    def timevec(self):
        dt,Nf = self.dt, self.length()
        return arange(0,Nf*dt, dt)[:Nf]

    def save_time_series_to_file(self, fname, ch, normp = False):
        rois = filter(self.isCircleROI, self.roi_objs.keys())
        ts = self.get_timeseries(normp=normp)
        t = self.timevec()        
        fd = file(fname, 'w')
        out_string = "Channel %d\n" % ch
        out_string += "Time\t" + '\t'.join(rois) + '\n'
        for k in xrange(self.length()):
            out_string += "%e\t"%t[k]
            out_string += '\t'.join(["%e"%a[k] for a in ts])
            out_string += '\n'
        fd.write(out_string)
        fd.close()


    def show_timeseries(self, rois = None, **keywords):
        t = self.timevec()
        for x,tag,roi,ax in self.roi_show_iterator(rois, **keywords):
            ax.plot(t, x, color = roi.get_facecolor())
            xlim((0,t[-1]))
    def show_ffts(self, rois = None, **keywords):
        L = self.length()
        freqs = fftfreq(int(L), self.dt)[1:L/2]
        for x,tag,roi,ax in self.roi_show_iterator(rois, **keywords):
            y = abs(fft(x))[1:L/2]
            ax.plot(freqs, y**2)
        ax.set_xlabel("Frequency, Hz")

    def show_spectrograms(self, rois = None, freqs = None,
                          wavelet = pycwt.Morlet(),
                          vmin = None,
                          vmax = None,
                          normp= True,
                          **keywords):
        keywords.update({'rois':rois, 'normp':normp})
        f_s = 1.0/self.dt
        freqs = ifnot(freqs, self.default_freqs())
        axlist = []
        for x,tag,roi,ax in self.roi_show_iterator(**keywords):
            aux.wavelet_specgram(x, f_s, freqs,  ax,
                                wavelet, vmin=vmin, vmax=vmax)
            axlist.append(ax)
        return axlist

    def show_spectrogram_with_ts(self, roilabel,
                                 freqs=None,
                                 wavelet = pycwt.Morlet(),
                                 title_string = None,
                                 vmin = None,
                                 vmax = None,
                                 normp = True,
                                 **keywords):
        "Create a figure of a signal, spectrogram and a colorbar"
        if not self.isCircleROI(roilabel):
            print "This is not a circle ROI, exiting"
            return
        signal = self.get_timeseries([roilabel],normp=normp)[0]
        Ns = len(signal)
        f_s = 1/self.dt
        freqs = ifnot(freqs,self.default_freqs())
        title_string = ifnot(title_string, roilabel)
        tvec = self.timevec()
        L = min(Ns,len(tvec))
        tvec,signal = tvec[:L],signal[:L]
        lc = self.roi_objs[roilabel].get_color()
        fig,axlist = aux.setup_axes1((8,4))
        axlist[1].plot(tvec, signal,'-',color=lc)
        axlist[1].set_xlabel('time, s')
        aux.wavelet_specgram(signal, f_s, freqs,  axlist[0], vmax=vmax,
                             wavelet=wavelet,
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
        ax.set_xlabel("Frequency, Hz")

    def default_freqs(self, nfreqs = 1024):
        return linspace(4.0/(self.length()*self.dt),
                      0.5/self.dt, num=nfreqs)

    def roi_show_iterator(self, rois = None,
                              normp=False):
            rois = ifnot(rois,
                         filter(self.isCircleROI, self.roi_objs.keys()))
            L = len(rois)
            if L < 1: return
            fig = figure(figsize=(8,4.5), dpi=80)
            for i, roi_label in enumerate(rois):
                roi = self.roi_objs[roi_label].obj
                ax = fig.add_subplot(L,1,i+1)
                x = self.roi_objs[roi_label].get_timeview(normp=normp)
                if i == L-1:
                    ax.set_xlabel("time, sec")
                ax.set_ylabel(roi_label)
                yield x, roi_label, roi, ax
            fig.show()


    def show_xwt_roi(self, tag1, tag2, freqs=None, ch=None,
                     func = pycwt.wtc_f,
                     wavelet = pycwt.Morlet()):
        "show cross wavelet spectrum or wavelet coherence for two ROIs"
        freqs = ifnot(freqs, self.default_freqs())
        self.extent=[0,self.length()*self.dt, freqs[0], freqs[-1]]

        if not (self.isCircleROI(tag1) and self.isCircleROI(tag2)):
            print "Both tags should be for CircleROIs"
            return

        s1 = self.roi_objs[tag1].get_timeview(True)
        s2 = self.roi_objs[tag2].get_timeview(True)

        res = func(s1,s2, freqs,1.0/self.dt,wavelet)

        t = self.timevec()

        figure();
        ax1= subplot(211);
        roi1,roi2 = self.roi_objs[tag1], self.roi_objs[tag2]
        plot(t,s1,color=roi1.get_color(), label=tag1)
        plot(t,s2,color=roi2.get_color(), label = tag2)
        legend()
        ax2 = subplot(212, sharex = ax1);
        ext = (t[0], t[-1], freqs[0], freqs[-1])
        ax2.imshow(res, extent = ext, cmap = aux.swanrgb())
        #self.cone_infl(freqs,wavelet)
        #self.confidence_contour(res,2.0)


    def show_xwt(self, **kwargs):
        for p in aux.allpairs0(filter(self.isCircleROI, self.roi_objs.keys())):
            self.show_xwt_roi(*p,**kwargs)


            
            
                
                    

            
        

        
