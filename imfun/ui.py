#!/usr/bin/python

import os
import sys
import glob
import time
import pylab as pl
import itertools as itt



import numpy as np
from pylab import mpl

from swan import pycwt

from imfun import lib, fnmap
ifnot = lib.ifnot
in_circle = lib.in_circle

from scipy import signal



mpl.rcParams['image.aspect'] = 'auto'
mpl.rcParams['image.origin'] = 'lower'


import itertools

def circle_from_struct(circ_props):
    cp = circ_props.copy()
    _  = cp.pop('func')
    center = cp.pop('center')
    return pl.Circle(center, **cp)

def line_from_struct(line_props):
    lp = line_props.copy()
    _ = lp.pop('func')
    xdata, ydata = lp.pop('xdata'), lp.pop('ydata')
    return LineScan(xdata, ydata,**lp)


import pickle
import random

vowels = "aeiouy"
consonants = "qwrtpsdfghjklzxcvbnm"


def rand_tag():
    return ''.join(map(random.choice,
                       (consonants, vowels, consonants)))
def tags_iter(tag_id =1):
    while True:
        yield 'r%02d'%tag_id
        tag_id +=1
        
def rand_tags_iter():
    while True: yield rand_tag()

def unique_tag(tags, max_tries = 1e4, tagger = tags_iter()):
    n = 0
    while n < max_tries:
        tag = tagger.next()
        n += 1
        if not tag in tags:
            return tag
    return "Err"


def nearest_item_ind(items, xy, fn = lambda a: a):
    """
    Index of the nearest item from a collection.
    Arguments: collection, position, selector
    """
    return lib.min1(lambda p: lib.eu_dist(fn(p), xy), items)




def line_from_points(p1,p2):
    p1 = map(float, p1)
    p2 = map(float, p2)
    k = (p1[1] - p2[1])/(p1[0] - p2[0])
    b = p1[1] - p1[0]*k
    return lambda x: k*x + b


def extract_line(data, xind, f):
    return np.array([data[int(i),int(f(i))] for i in xind])


def extract_line2(data, p1, p2):
    L = int(lib.eu_dist(p1,p2))
    f = lambda x1,x2: lambda i: int(x1 + i*(x2-x1)/L)
    return np.array([data[f(p1[1],p2[1])(i), f(p1[0],p2[0])(i)]
                     for i in range(L)])


def color_walker():
    ar1 = lib.ar1
    red, green, blue = ar1(), ar1(), ar1()
    while True:
        yield map(lambda x: np.mod(x.next(),1.0), (red,green,blue))



def rezip(a):
    return zip(*a)


def view_fseq_frames(fseq, vmin = None, vmax = None):
    f = pl.figure()
    axf = pl.axes()
    frame_index = [0]
    frames = fseq.as3darray()
    Nf = len(frames)

    if vmax is None:
        vmax = np.max(map(np.max, frames))

    if vmin is None:
        vmin = np.min(map(np.min, frames))


    sy,sx = fseq.shape()
    dy,dx, scale_setp = fseq.get_scale()

    plf = axf.imshow(frames[0],
                     extent = (0, sx*dx, 0, sy*dy),
                     interpolation = 'nearest',
                     vmax = vmax, vmin = vmin,
                     aspect = 'equal', cmap=mpl.cm.gray)
    if scale_setp:
        pl.ylabel('um')
        pl.xlabel('um')
    pl.colorbar(plf)
    def skip(event,n=1):
        fi = frame_index[0]
        key = hasattr(event, 'button') and event.button or event.key
        if key in (4,'4','down','left'):
            fi -= n
        elif key in (5,'5','up','right'):
            fi += n
        fi = fi%Nf
        plf.set_data(frames[fi])
        axf.set_title('%03d (%3.3f sec)'%(fi, fi*fseq.dt))
        frame_index[0] = fi
        f.canvas.draw()
    f.canvas.mpl_connect('scroll_event',skip)
    #f.canvas.mpl_connect('key_press_event', skip)
    f.canvas.mpl_connect('key_press_event',lambda e: skip(e,10))


class DraggableObj:
    verbose = True
    def __init__(self, obj, parent,):
        self.obj = obj
        self.parent = parent
        self.connect()
        self.pressed = None
        self.tag = obj.get_label() # obj must support this
        fseq = self.parent.fseq
        self.dy,self.dx, self.scale_setp = fseq.get_scale()
        self.set_tagtext()
        return

    def redraw(self):
        self.obj.axes.figure.canvas.draw()

    def event_ok(self, event, should_contain=False):
        containsp = True
        if should_contain:
            containsp, _ = self.obj.contains(event)
        return event.inaxes == self.obj.axes and \
               containsp and \
               pl.get_current_fig_manager().toolbar.mode ==''

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
    def set_tagtext(self):
        pass

    def on_scroll(self, event): pass

    def on_type(self, event): pass
    
    def on_press(self, event):
        pass

    def disconnect(self):
        map(self.obj.axes.figure.canvas.mpl_disconnect,
            self.cid.values())
    def get_color(self):
        return self.obj.get_facecolor()


class LineScan(DraggableObj):
    def __init__(self, obj, parent):
        DraggableObj.__init__(self, obj, parent)
        ep = self.endpoints()[1]
        ax = self.obj.axes
        self.length_tag = ax.text(ep[0],ep[1], '%2.2f'%self.length(),
                                  size = 'small',
                                  color = self.obj.get_color())
        print "%2.2f"%self.length()
        self.parent.legend()
        self.redraw()
    def update_length_tag(self):
        "updates text with line length"
        lt = self.length_tag
        ep = self.endpoints()[1]
        lt.set_position(ep)
        lt.set_text('%2.2f'%self.length())
    def endpoints(self):
        return rezip(self.obj.get_data())
    def length(self):
        return lib.eu_dist(*self.endpoints())
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
        dist1,dist2 = [lib.eu_dist(p,x) for x in self.endpoints()]
        if dist1/(dist1 + dist2) < 0.05:
            dx,dy = np.array([dx, 0]), np.array([dy, 0])
        elif dist2/(dist1 + dist2) < 0.05:
            dx,dy = np.array([0, dx]), np.array([0, dy])
        self.obj.set_data((x0 + dx,y0 + dy))
        self.pressed = p
        self.update_length_tag()

    def on_press(self, event):
        if not self.event_ok(event, True):
            return
        if event.button == 3:
            self.disconnect()
            self.length_tag.set_alpha(0)
            self.length_tag.remove()
            self.obj.remove()
            self.parent.roi_objs.pop(self.tag)
            self.parent.legend()
            self.redraw()
        elif event.button == 1:
            x, y = self.obj.get_xdata(), self.obj.get_ydata()
            x0 = x[1] - x[0]
            y0 = y[1] - y[0]
            self.pressed = event.xdata, event.ydata
        elif event.button == 2:
            self.show_timeview()

    def transform_point(self, p):
        return p[0]/self.dx, p[1]/self.dy

    def get_timeview(self):
        points = map(self.transform_point, self.check_endpoints())
        timeview = np.array([extract_line2(frame, *points) for frame in
                             self.parent.fseq.frames()])
        return timeview,points

    def show_timeview(self):
        timeview,points = self.get_timeview()
        if timeview is not None:
            fseq = self.parent.fseq
            ax = pl.figure().add_subplot(111)
            # TODO: work out if sx differs from sy
            ax.imshow(timeview,
                      extent=(0, self.dx*timeview.shape[1], 0,
                              fseq.dt*self.parent.length()),
                      )
                      #aspect='equal')
            if self.scale_setp:
                ax.set_xlabel('um')
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
        self.set_tagtext()
        self.redraw()
        return

    def on_type(self, event):
        if not self.event_ok(event, True): return
        if self.verbose:
            print event.key
        tags = [self.tag]
        if event.key in ['t', '1']:
            self.parent.show_timeseries(tags)
        elif event.key in ['c']:
            self.parent.show_xcorrmap(self.tag)
        elif event.key in ['T', '!']:
            self.parent.show_timeseries(tags, normp=True)
        elif event.key in ['w', '2']:
            self.parent.show_spectrogram_with_ts(self.tag)
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
            self.destroy()
        
    def destroy(self):
        p = self.parent.roi_objs.pop(self.tag)
        self.obj.remove()
        self.tagtext.remove()
        self.disconnect()
        #self.obj.axes.legend()
        #self.parent.legend()
        self.redraw()

    def move(self, p):
        "Move the ROI when the mouse is pressed over it"
        xp,yp, x, y = self.pressed
        dx = p[0] - xp
        dy = p[1] - yp
        self.obj.center = (x+dx, y+dy)
        self.set_tagtext()
        self.redraw()

    def set_tagtext(self):
        ax = self.obj.axes
        p = self.obj.center
        r = self.obj.get_radius()
        x = p[0] + np.sin(np.pi/4)*r
        y = p[1] + np.sin(np.pi/4)*r
        if not hasattr(self, 'tagtext'):
            self.tagtext = ax.text(x,y, '%s'%self.obj.get_label(),
                                   size = 'small',
                                   color = self.get_color())
        else:
            self.tagtext.set_position((x,y))

    def in_circle(self, shape):
        roi = self.obj
        c = roi.center[0]/self.dx, roi.center[1]/self.dy
        fn = lib.in_circle(c, roi.radius/self.dx)
        X,Y = np.meshgrid(*map(range, shape[::-1]))
        return fn(X,Y)

    def get_timeview(self, normp=False):
        shape = self.parent.fseq.shape()
        v = self.parent.fseq.mask_reduce(self.in_circle(shape))
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


class DragRect(DraggableObj):
    def __init__(self, obj):
        self.obj = obj
        self.connect()
        self.pressed = None
        self.tag = obj.get_label() # obj must support this
        pass

    "Draggable Rectangular ROI"
    def on_press(self, event):
        if not self.event_ok(event, True): return
        x0, y0 = self.obj.xy
        self.pressed = x0, y0, event.xdata, event.ydata
    def move(self, p):
        x0, y0, xpress, ypress = self.pressed
        dx = p[0] - xpress
        dy = p[1] - ypress
        self.obj.set_x(x0+dx)
        self.obj.set_y(y0+dy)
    def jump(self,p):
        self.obj.set_x(p[0])
        self.obj.set_y(p[1])

def intens_measure(arr1, arr2):
    pass

def sse_measure(arr1,arr2, vrange=255.0):
    r,c = arr1.shape
    max_sse = r*c*vrange
    return 1 - np.sqrt(np.sum((arr1-arr2)**2))/max_sse

def corr_measure(arr1,arr2):
    return 0.5 + np.sum(arr1*arr2)/(2*float(arr1.std())*float(arr2.std()))


class RectFollower(DragRect):
    def __init__(self, obj, arr):
        DragRect.__init__(self, obj)
        self.search_width = 1.5*obj.get_width()
        self.search_height = obj.get_width()
        self.arr = arr
    def on_type(self, event):
        if not self.event_ok(event, True): return
    def xy(self):
        return map(int, self.obj.get_xy())
    def toslice(self,xoff=0,yoff=0):
        start2,start1 = self.xy()
        start2 += xoff
        start1 += yoff
        w, h = self.obj.get_width(), self.obj.get_height()
        return (slice(start1, start1+h), slice(start2, start2+w))
    def find_best_pos(self, frame, template, measure=sse_measure):
        acc = {}
        sw, sh = self.search_width, self.search_height
        _,limh, limw = self.arr.shape
        for w in range(max(0,-sw/2), min(sw/2,limw)):
            for h in range(max(0,-sh/2), min(limh,sh/2)):
                s = self.toslice(h,w)
                d = measure(frame[s], template)
                acc[(w,h)] = d
        pos = sorted(acc.items(), lambda x, y: cmp(x[1], y[1]), reverse=True)
        pos = pos[0][0]
        x,y = self.xy()
        return x+pos[0], y+pos[1]
        
        
        
def synthetic_vessel(nframes, width = 80, shape=(512,512), noise = 0.5):
    z = lambda : np.zeros(shape)
    left = 100
    right = left+width
    frames = []
    xw1 = lib.ar1()
    xw2 = lib.ar1()
    for i in range(nframes):
        f = np.zeros(shape)
        l,r= 2*left+xw1.next(),2*right+xw2.next()
        f[:,l:l+4] = 1.0
        f[:,r:r+4] = 1.0
        f += np.random.randn(*shape)*noise
        frames.append(((f.max()-f)/f.max())*255.0)
    return np.array(frames)
        
    

def track_vessels(frames, width=30, height=60, measure = sse_measure):
    f = pl.figure()
    axf = pl.axes()
    frame_index = [0]
    Nf = len(frames)

    plf = axf.imshow(frames[0],
                     interpolation = 'nearest',
                     aspect = 'equal', cmap=mpl.cm.gray)
    pl.colorbar(plf)
    def skip(event,n=1):
        fi = frame_index[0]
        key = hasattr(event, 'button') and event.button or event.key
        k = 1
        s1,s2 = [r.toslice() for r in drs]
        tmpl1,tmpl2 = [frames[fi][s] for s in s1,s2]

        if key in (4,'4','down','left'):
            k = -1
        elif key in (5,'5','up','right'):
            k = 1
        fi += k*n
        fi = fi%Nf
        plf.set_data(frames[fi])
        new_pos1 = drs[0].find_best_pos(frames[fi],tmpl1,measure)
        new_pos2 = drs[1].find_best_pos(frames[fi],tmpl2,measure)
        drs[0].jump(new_pos1)
        drs[1].jump(new_pos2)
        axf.set_title('frame %03d, p1 %f, p2 %f)'%(fi, drs[0].obj.get_x(), drs[1].obj.get_x()))
        frame_index[0] = fi
        f.canvas.draw()
    f.canvas.mpl_connect('scroll_event',skip)
    #f.canvas.mpl_connect('key_press_event', skip)
    #f.canvas.mpl_connect('key_press_event',lambda e: skip(e,10))
    rects = axf.bar([0, 60], [height, height], [width,width], alpha=0.2)
    drs = [RectFollower(r,frames) for r in rects]
    f.canvas.draw()
    return drs



class Picker:
    verbose = True
    def __init__(self, fseq):
        self.cw = color_walker()

        self._show_legend=False
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
    
    def on_click(self,event):
        tb = pl.get_current_fig_manager().toolbar
        if event.inaxes != self.ax1 or tb.mode != '': return

        x,y = round(event.xdata), round(event.ydata)
        if event.button is 1 and \
           not self.any_roi_contains(event):
            label = unique_tag(self.roi_tags(), tagger=self.tagger)
            c = pl.Circle((x,y), 5, alpha = 0.5,
                       label = label,
                       color=self.cw.next())
            c.figure = self.fig
            self.ax1.add_patch(c)
            self.roi_objs[label]= CircleROI(c, self)
            #drc.connect()
        self.legend()    
        #self.ax1.legend()
        pl.draw()

    def legend(self):
        if self._show_legend == True:
            keys = sorted(self.roi_objs.keys())
            handles = [self.roi_objs[key].obj for key in keys]
            try:
                axs= self.ax1.axis
                if self.legtype is 'figlegend':
                    pl.figlegend(handles, keys, 'upper right')
                elif self.legtype is 'axlegend':
                    self.ax1.legend(handles, keys)
                    self.ax1.axis(axs)
                    self.redraw()
            except Exception as e:
                    print "Picker: can't make legend because ", e
    def on_press(self, event):
        if event.inaxes !=self.ax1 or \
               self.any_roi_contains(event) or \
               event.button != 3 or \
               pl.get_current_fig_manager().toolbar.mode !='':
            return
        self.pressed = event.xdata, event.ydata
        axrange = self.ax1.get_xbound() + self.ax1.get_ybound()
        self.curr_line_handle = self.init_line_handle()
        self.ax1.axis(axrange)
        return
    def init_line_handle(self):
        lh, = self.ax1.plot([0],[0],'-', color=self.cw.next())
        return lh


    def on_motion(self, event):
        if (self.pressed is None) or (event.inaxes != self.ax1):
            return
        pstop = event.xdata, event.ydata
        if hasattr(self, 'curr_line_handle'): 
            self.curr_line_handle.set_data(*rezip((self.pressed,pstop)))
        self.fig.canvas.draw() #todo BLIT!
        
    def on_release(self,event):
        self.pressed = None
        if not event.button == 3: return
        if self.any_roi_contains(event): return
        if not hasattr(self, 'curr_line_handle'): return
        if len(self.curr_line_handle.get_xdata()) > 1:
            tag = unique_tag(self.roi_tags(), tagger=self.tagger)
            self.curr_line_handle.set_label(tag)
            newline = LineScan(self.curr_line_handle, self)
            if newline.length() > self.min_length:
                self.roi_objs[tag] = newline
            else:
                try:
                    self.curr_line_handle.remove()
                except: pass
        else:
            try:
                self.curr_line_handle.remove()
            except Exception as e:
                print "Can't remove line handle because", e
        self.curr_line_handle = self.init_line_handle()
        self.legend()
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
        print "Saving ROIs to ", fname
        if fname:
            pickle.dump([x.to_struct() for x in self.roi_objs.values()],
                        file(fname, 'w'),
                        protocol=0)
#                     if isinstance(x, CircleROI)],


    def load_rois(self, fname):
        "Load stored ROIs from a file"
        saved_rois = pickle.load(file(fname))
        rois = [x['func'](x) for x in saved_rois]
        circles = filter(lambda x: isinstance(x, pl.Circle), rois)
        lines = filter(lambda x: isinstance(x, LineScan), rois)
        map(self.ax1.add_patch, circles) # add points to the axes
        map(self.ax1.add_line, lines) # add points to the axes
        self.roi_objs.update(dict([(c.get_label(), CircleROI(c,self))
                                   for c in circles]))
        self.roi_objs.update(dict([(l.get_label(), LineScan(l,self))
                                   for l in lines]))

        #self.ax1.legend()
        pl.draw() # redraw the axes
        return

    def start(self, roi_objs={}, ax=None, legend_type = 'figlegend',
              mean_frame =True,
              vmax = None, vmin = None, 
              cmap = 'gray',
              interpolation = 'nearest'):
        "Start picking up ROIs"
        self.tagger = tags_iter()
        self.drcs = {}
        self.ax1 = ifnot(ax, pl.figure().add_subplot(111))
        self.fig = self.ax1.figure
        self.legtype = legend_type
        self.pressed = None

        dx,dy, scale_setp = self.fseq.get_scale()
        sy,sx = self.fseq.shape()
        avmin,avmax = self.fseq.data_range()
        if vmin is None: vmin = avmin
        if vmax is None: vmax = avmax
        if hasattr(self.fseq, 'ch'):
            pl.title("Channel: %s" % ('red', 'green')[self.fseq.ch] )
        if mean_frame:
            f = self.fseq.mean_frame()
        else:
            f = self.fseq.frames().next()
        self.pl = self.ax1.imshow(f,
                                  extent = (0, sx*dx, 0, sy*dy),
                                  interpolation = interpolation,
                                  aspect='equal',
                                  vmax=vmax,  vmin=vmin,
                                  cmap=cmap)
        if scale_setp:
            self.ax1.set_xlabel('um')
            self.ax1.set_ylabel('um')

        self.disconnect()
        self.connect()

        return self.ax1, self.pl

    def connect(self):
        "connect all the needed events"
        print "connecting callbacks to picker"
        cf = self.fig.canvas.mpl_connect
        self.cid = {
            'click': cf('button_press_event', self.on_click),
            'press': cf('button_press_event', self.on_press),
            'release': cf('button_release_event', self.on_release),
            'motion': cf('motion_notify_event', self.on_motion),
            }
    def disconnect(self):
        if hasattr(self, 'cid'):
            print "disconnecting old callbacks"
            map(self.fig.canvas.mpl_disconnect, self.cid.values())
            
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
        return np.arange(0,Nf*dt, dt)[:Nf]

    def save_time_series_to_file(self, fname, normp = False):
        rois = filter(self.isCircleROI, self.roi_objs.keys())
        ts = self.get_timeseries(normp=normp)
        t = self.timevec()        
        fd = file(fname, 'w')
        if hasattr(self.fseq, 'ch'):
            out_string = "Channel %d\n" % self.fseq.ch
        else:
            out_string = ""
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
            pl.xlim((0,t[-1]))
    def show_ffts(self, rois = None, **keywords):
        L = self.length()
        freqs = np.fft.fftfreq(int(L), self.dt)[1:L/2]
        for x,tag,roi,ax in self.roi_show_iterator(rois, **keywords):
            y = abs(np.fft.fft(x))[1:L/2]
            ax.plot(freqs, y**2)
        ax.set_xlabel("Frequency, Hz")
    def show_xcorrmap(self, roitag, figsize=(6,6),
                      **kwargs):
        from scipy import ndimage
        roi =  self.roi_objs[roitag]
        signal = self.get_timeseries([roitag],normp=True)[0]
        xcmap = fnmap.xcorrmap(self.fseq, signal, corrfn='pearson',**kwargs)
        mask = roi.in_circle(xcmap.shape)
        xshow = np.ma.masked_where(mask,xcmap)
        fig = pl.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        vmin, vmax = xshow.min(), xshow.max()
        im = ax.imshow(ndimage.median_filter(xcmap,3), aspect = 'equal', vmin=vmin,
                       vmax=vmax,cmap=lib.swanrgb)
        pl.colorbar(im, ax=ax)
        ax.set_title('Correlation to %s'%roitag)
        return xcmap

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
            lib.wavelet_specgram(x, f_s, freqs,  ax,
                                wavelet, vmin=vmin, vmax=vmax)
            axlist.append(ax)
        return axlist

    def show_spectrogram_with_ts(self, roitag,
                                 freqs=None,
                                 wavelet = pycwt.Morlet(),
                                 title_string = None,
                                 vmin = None,
                                 vmax = None,
                                 normp = True,
                                 **keywords):
        "Create a figure of a signal, spectrogram and a colorbar"
        if not self.isCircleROI(roitag):
            print "This is not a circle ROI, exiting"
            return
        signal = self.get_timeseries([roitag],normp=normp)[0]
        Ns = len(signal)
        f_s = 1/self.dt
        freqs = ifnot(freqs,self.default_freqs())
        title_string = ifnot(title_string, roitag)
        tvec = self.timevec()
        L = min(Ns,len(tvec))
        tvec,signal = tvec[:L],signal[:L]
        lc = self.roi_objs[roitag].get_color()
        fig,axlist = lib.setup_axes_for_spectrogram((8,4))
        axlist[1].plot(tvec, signal,'-',color=lc)
        axlist[1].set_xlabel('time, s')
        lib.wavelet_specgram(signal, f_s, freqs,  axlist[0], vmax=vmax,
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
        return np.linspace(4.0/(self.length()*self.dt),
                           0.5/self.dt, num=nfreqs)

    def roi_show_iterator(self, rois = None,
                              normp=False):
            rois = ifnot(rois,
                         filter(self.isCircleROI, self.roi_objs.keys()))
            L = len(rois)
            if L < 1: return
            fig = pl.figure(figsize=(8,4.5), dpi=80)
            for i, roi_label in enumerate(rois):
                roi = self.roi_objs[roi_label].obj
                ax = fig.add_subplot(L,1,i+1)
                x = self.roi_objs[roi_label].get_timeview(normp=normp)
                if i == L-1:
                    ax.set_xlabel("time, sec")
                ax.set_ylabel(roi_label)
                yield x, roi_label, roi, ax
            fig.show()


    def show_xwt_roi(self, tag1, tag2, freqs=None,
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

        pl.figure();
        ax1= pl.subplot(211);
        roi1,roi2 = self.roi_objs[tag1], self.roi_objs[tag2]
        pl.plot(t,s1,color=roi1.get_color(), label=tag1)
        pl.plot(t,s2,color=roi2.get_color(), label = tag2)
        #legend()
        ax2 = pl.subplot(212, sharex = ax1);
        ext = (t[0], t[-1], freqs[0], freqs[-1])
        ax2.imshow(res, extent = ext, cmap = lib.swanrgb)
        #self.cone_infl(freqs,wavelet)
        #self.confidence_contour(res,2.0)


    def show_xwt(self, **kwargs):
        for p in lib.allpairs0(filter(self.isCircleROI, self.roi_objs.keys())):
            self.show_xwt_roi(*p,**kwargs)


            
            
                
                    

            
        

        
