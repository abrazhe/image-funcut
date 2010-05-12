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
    prev = randn()
    while True:
        res = prev*alpha + randn()
        prev = res
        yield res


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


class DraggableObj():
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
    def on_scroll(self, event):
        pass
    def on_motion(self, event):
        if not (self.event_ok(event, False) and self.pressed):
            return
        p = event.xdata,event.ydata
        self.move(p)
        #self.pressed = p
        self.redraw()

    def on_release(self, event):
        if not self.event_ok(event):
            return
        self.pressed = None
        self.redraw()

    def on_type(self, event):
        pass
    
    def on_press(self, event):
        pass

    def disconnect(self):
        map(self.obj.axes.figure.canvas.mpl_disconnect,
            self.cid.values())
    def get_color(self):
        return self.obj.get_facecolor()


class DraggableLine(DraggableObj):
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
    def move(self,p):
        # xp,yp, x, y = self.pressed
        xp,yp   = self.pressed
        dx = p[0] - xp
        dy = p[1] - yp
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
            self.parent.lines.pop(self.tag) 
            self.disconnect()
            self.obj.remove()
            self.redraw()
        elif event.button == 1:
            x, y = self.obj.get_xdata(), self.obj.get_ydata()
            x0 = x[1] - x[0]
            y0 = y[1] - y[0]
            self.pressed = event.xdata, event.ydata
        elif event.button == 2:
            self.parent.show_timeview(self.tag)

class DraggableCircle(DraggableObj):
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
            p = self.parent.drcs.pop(self.tag)
            self.obj.remove()
            self.disconnect()
            legend()
            self.redraw()

    def move(self, p):
        "Move the ROI if the mouse is over it"
        xp,yp, x, y = self.pressed
        dx = p[0] - xp
        dy = p[1] - yp
        self.obj.center = (x+dx, y+dy)
        self.redraw()

     


class ImgLineScan():
    verbose=True
    connected = False
    cw = color_walker()
    def __init__(self, fseq, min_length=5):
        self.fseq = fseq
        self.dt = fseq.dt
        self._Nf = None
        self.lines = {}
        self.min_length = min_length
        pass

    def length(self):
        if self._Nf is None:
            self._Nf = self.fseq.length()
        return self._Nf
    def any_line_contains(self,event):
        "Checks if event is contained by any ROI"
        if len(self.lines) < 1 : return False
        return reduce(lambda x,y: x or y,
                      [line.obj.contains(event)[0]
                       for line in self.lines.values()])
    def pick_lines(self,lines = {}):
        self.fig = pl.figure()
        self.ax1 = self.fig.add_subplot(111)
        if hasattr(self.fseq, 'ch'):
            title("Channel: %s" % ('red', 'green')[self.fseq.ch] )
        self.image = self.ax1.imshow(self.fseq.mean_frame(),
                                     aspect='equal',
                                     cmap=matplotlib.cm.gray)
        if self.connected is False:
            self.pressed = None
            self.fig.canvas.mpl_connect('button_press_event',self.on_press)
            self.fig.canvas.mpl_connect('button_release_event',self.on_release)
            self.fig.canvas.mpl_connect('motion_notify_event',self.on_motion)
            self.connected = True
        pass

    def line_tags(self):
        return self.lines.keys()

    def on_press(self, event):
        if event.inaxes !=self.ax1 or \
               self.any_line_contains(event) or \
               event.button != 1 or \
               get_current_fig_manager().toolbar.mode !='':
            return
        self.pressed = event.xdata, event.ydata
        axrange = self.ax1.get_xbound() + self.ax1.get_ybound()
        tag = unique_tag(self.line_tags())
        self.curr_line_handle, = self.ax1.plot([0],[0],'-',
                                               label = tag,
                                               color=self.cw.next())
        pl.axis(axrange)
        return

    def on_motion(self, event):
        if (self.pressed is None) or (event.inaxes != self.ax1):
            return
        pstop = event.xdata, event.ydata
        self.curr_line_handle.set_data(*rezip((self.pressed,pstop)))
        self.fig.canvas.draw() #todo BLIT!
        
    def on_release(self,event):
        self.pressed = None
        if not event.button == 1: return
        if self.any_line_contains(event): return
        tag = self.curr_line_handle.get_label()
        if len(self.curr_line_handle.get_xdata()) > 1:
            newline = DraggableLine(self.curr_line_handle, self)
            if newline.length() > self.min_length:
                self.lines[tag] = newline
            else:
                self.curr_line_handle.remove()
        else:
            self.curr_line_handle.remove()
        if len(self.lines) > 0:
            pl.legend()
        self.fig.canvas.draw() #todo BLIT!
        return

    def get_timeview(self, tag):
        if not tag in self.lines.keys():
            print("Sorry, no line with this tag")
            return None
        line = self.lines[tag]
        points = line.check_endpoints()
        timeview = array([extract_line2(frame, *points) for frame in
                          self.fseq.frames()])
        return timeview, points

    def show_timeview(self,tag):
        timeview, points = self.get_timeview(tag)
        if timeview is not None:
            pl.figure();
            pl.imshow(timeview,
                      extent=(0, timeview.shape[1], 0, self.fseq.dt*self.length()),
                      aspect='equal')
            pl.ylabel('time, sec')
            pl.title('Timeview for '+ tag)


class ImgPointSelect():
    verbose = True
    connected = False
    cw = color_walker()

    def __init__(self, fseq):
        self.fseq = fseq
        self.dt = fseq.dt
        self._Nf = None
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
            
            label = unique_tag(self.roi_labels())
            c = Circle((x,y), 5, alpha = 0.5,
                       label = label,
                       color=self.cw.next())
            c.figure = self.fig
            self.ax1.add_patch(c)
            print c.axes
            self.drcs[label]= DraggableCircle(c, self)
            #drc.connect()
            
        legend()
        draw()

    def any_roi_contains(self,event):
        "Checks if event is contained by any ROI"
        if len(self.drcs) < 1 : return False
        return reduce(lambda x,y: x or y,
                      [roi.obj.contains(event)[0]
                       for roi in self.drcs.values()])
    
    def roi_labels(self):
        "List of labels for all ROIs"
        return self.drcs.keys()
    
    def save_rois(self, fname):
        "Saves picked ROIs to a file"
        pickle.dump(map(struct_circle,
                        [x.obj for x in self.drcs.values()]),
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

    def pick_rois(self, points = []):
        "Start picking up ROIs"
        self.drcs = {}
        self.fig = figure()
        self.ax1 = axes()
        if hasattr(self.fseq, 'ch'):
            title("Channel: %s" % ('red', 'green')[self.fseq.ch] )
        self.pl = self.ax1.imshow(self.fseq.mean_frame(),
                                  aspect='equal',
                                  cmap=matplotlib.cm.gray)
        if True or self.connected is False:
            self.fig.canvas.mpl_connect('button_press_event',
                                        self.onclick)
            #self.fig.canvas.mpl_connect('pick_event', self.onpick)
            self.connected = True

    def roi_timeview(self, tag, normp=False):
        roi = self.drcs[tag].obj
        fn = in_circle(roi.center, roi.radius)
        shape = self.fseq.shape()
        X,Y = meshgrid(*map(range, shape))
        v = self.fseq.mask_reduce(fn(X,Y))
        if normp:
            Lnorm = type(normp) is int and normp or len(v)
            return (v-np.mean(v[:Lnorm]))/np.std(v[:Lnorm])
        else: return v
        return 
        
    def list_roi_timeseries_from_labels(self, roi_labels, **keywords):
        "Returns timeseres for a list of roi labels"
        return [self.roi_timeseries_from_label(label, **keywords)
                for label in roi_labels]
    
    def get_timeseries(self, rois=None, normp=False):
        rois = ifnot(rois, self.drcs.keys())
        return [self.roi_timeview(r, normp)
                for r in  rois]

    def timevec(self):
        dt,Nf = self.dt, self.length()
        return arange(0,Nf*dt, dt)[:Nf]

    def save_time_series_to_file(self, fname, ch, normp = False):
        rois = self.drcs.keys()
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
    
    def show_spectrogram_with_ts(self, roilabel,
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
        freqs = ifnot(freqs,self.default_freqs())
        title_string = ifnot(title_string, roilabel)
        tvec = self.timevec()
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
        ax.set_xlabel("Frequency, Hz")

    def default_freqs(self, nfreqs = 1024):
        return linspace(4.0/(self.length()*self.dt),
                      0.5/self.dt, num=nfreqs)

    def roi_show_iterator(self, rois = None,
                              normp=False):
            rois = ifnot(rois, self.drcs.keys())
            L = len(rois)
            if L < 1: return
            fig = figure(figsize=(8,4.5), dpi=80)
            for i, roi_label in enumerate(rois):
                roi = self.drcs[roi_label].obj
                ax = fig.add_subplot(L,1,i+1)
                x = self.roi_timeview(roi_label, normp=normp)
                if i == L-1:
                    ax.set_xlabel("time, sec")
                ax.set_ylabel(roi_label)
                yield x, roi_label, roi, ax
            fig.show()

    def wfnmap(self,extent, nfreqs = 16,
               wavelet = pycwt.Morlet(),
               func = np.mean,
               normL = None,
               kern=None,
               **kwargs):
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

        *kern* -- if 0, no alias, then each frame is filtered, if an array,
        use this as a kernel to convolve each frame with; see aliased_pix_iter
        for default kernel
        """
        tick = time.clock()
        
        shape = self.fseq.shape(kwargs.has_key('sliceobj') and
                                kwargs['sliceobj'] or None)
        out = ones(shape, np.float64)
        total = shape[0]*shape[1]
        k = 0
        freqs = linspace(*extent[2:], num=nfreqs)
        pix_iter = None
        normL = ifnot(normL, self.length())

        if type(kern) == np.ndarray or kern is None:
            pix_iter = self.fseq.conv_pix_iter(kern,**kwargs)
        elif kern <= 0:
            pix_iter = self.pix_iter(**kwargs)
        
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

    def show_xwt_roi(self, tag1, tag2, freqs=None, ch=None,
                     func = pycwt.wtc_f,
                     wavelet = pycwt.Morlet()):
        "show cross wavelet spectrum or wavelet coherence for two ROI"
        freqs = ifnot(freqs, self.default_freqs())
        self.extent=[0,self.length()*self.dt, freqs[0], freqs[-1]]

        s1 = self.roi_timeview(tag1,True)
        s2 = self.roi_timeview(tag2,True)
        res = func(s1,s2, freqs,1.0/self.dt,wavelet)

        t = self.timevec()

        figure();
        ax1= subplot(211);
        roi1,roi2 = self.drcs[tag1], self.drcs[tag2]
        plot(t,s1,color=roi1.get_color(), label=tag1)
        plot(t,s2,color=roi2.get_color(), label = tag2)
        legend()
        ax2 = subplot(212, sharex = ax1);
        ext = (t[0], t[-1], freqs[0], freqs[-1])
        ax2.imshow(res, extent = ext, cmap = aux.swanrgb())
        #self.cone_infl(freqs,wavelet)
        #self.confidence_contour(res,2.0)

    def ffnmap(self, fspan, kern = None, func=np.mean):
        """
        Fourier-based functional mapping
        fspan : a range of frequencies in Hz, e.g. (1.0, 1.5)
        kern  : a kernel to convolve each frame with
        func  : range reducing function. np.mean by default, may be np.sum as well
        """
        tick = time.clock()
        shape = self.fseq.shape()
        out = ones(shape, np.float64)
        total = shape[0]*shape[1]
        k = 0
        freqs = fftfreq(self.length(), self.dt)
        pix_iter = self.fseq.conv_pix_iter(kern)
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
        for p in aux.allpairs0(self.drcs.keys()):
            self.show_xwt_roi(*p,**kwargs)


            
            
                
                    

            
        

        
