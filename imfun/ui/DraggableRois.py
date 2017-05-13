 # a/b will always return float

import numpy as np

from scipy import ndimage
from scipy.interpolate import splev, splprep

from matplotlib import pyplot as plt
from matplotlib import widgets as mw
import matplotlib as mpl

from .. import core
from .. import track
from ..core import coords
from ..core import ifnot, rezip

from ..core.baselines import DFoSD, DFoF
import collections


in_circle = coords.in_circle

def circle_from_struct(circ_props):
    cp = circ_props.copy()
    cp.pop('func')
    center = cp.pop('center')
    return plt.Circle(center, **cp)

def line_from_struct(line_props):
    lp = line_props.copy()
    lp.pop('func')
    xdata, ydata = lp.pop('xdata'), lp.pop('ydata')
    lh = plt.Line2D(xdata, ydata, **lp)
    return lh

def line_from_points(p1,p2):
    k = (p1[1] - p2[1])/(p1[0] - p2[0])
    b = p1[1] - p1[0]*k
    return k, b


def rot_vector(v, alpha):
    sina,cosa = np.sin(alpha),np.cos(alpha)
    return np.dot([[cosa, -sina], [sina, cosa]], v)

def translate(p1,p2):
    """returns translation vector used to
    ||-translate line segment p1,p2 in an orthogonal direction"""
    L = coords.eu_dist(p1, p2)
    v = [np.array(p, ndmin=2).T for p in (p1,p2)]
    T = np.array([[0, -1], [1, 0]]) # basic || translation vector
    tv = np.dot(T, v[0] - v[1])/L
    return tv

def line_reslice1(data, xind, f):
    return np.array([data[int(i),int(f(i))] for i in xind])


def line_reslice2(data, p1, p2):
    L = int(coords.eu_dist(p1,p2))
    #f = lambda x1,x2: lambda i: int(x1 + i*(x2-x1)/L)
    def f(x1,x2): return lambda i: int(x1 + i*(x2-x1)/L)
    return np.array([data[f(p1[1],p2[1])(i), f(p1[0],p2[0])(i)]
                     for i in range(L)])

def line_reslice3(data, p1,p2,order=1):
    #
    L = int(round(coords.eu_dist(p1,p2)))
    #uv = (p2-p1)/L
    uv = np.diff((p1,p2),axis=0)/L
    coords_ =  p1 + uv*np.arange(L+1)[:,None]
    #print coords.shape, data.shape
    return ndimage.map_coordinates(data, coords_.T[::-1],order=order)


#def rezip(a):
#    return zip(*a)


## def resample_velocity(rad, v,rstart=10.0):
##     rstop = rad[-1]
##     xnew = np.arange(rstart, rstop, 0.25)
##     return xnew, np.interp(xnew,rad,v)

class DraggableObj(object):
    """Basic class for objects that can be dragged on the screen"""
    verbose = True
    def __init__(self, obj, parent,):
        self.obj = obj # matplotlib object
        self.parent = parent
        self.connect()
        self.pressed = None
        self.tag = obj.get_label() # obj must support this
        if hasattr(parent, 'frame_coll'):
            frame_coll = self.parent.frame_coll
            self.axes = frame_coll.meta['axes']
        self.set_tagtext()
        self.canvas = obj.axes.figure.canvas
        return

    def redraw(self):
        self.canvas.draw()

    def event_ok(self, event, should_contain=False):
        containsp = True
        if should_contain:
            containsp, _ = self.obj.contains(event)
        return event.inaxes == self.obj.axes and \
               containsp and \
               self.obj.axes.figure.canvas.toolbar.mode ==''

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
        self.pressed = None
        if not self.event_ok(event):
            return
        self.redraw()

    def set_tagtext(self):
        pass

    def on_scroll(self, event): pass
    def on_type(self, event): pass
    def on_press(self, event):   pass

    def disconnect(self):
        for tag in self.cid:
            self.obj.axes.figure.canvas.mpl_disconnect(self.cid[tag])


    @property
    def color(self):
        return self.obj.get_edgecolor()

class ThresholdLine(DraggableObj):
    def on_press(self,event):
        if self.event_ok(event, True):
            self.pressed = event.xdata,event.ydata
    def move(self,p):
        self.set_value(p[1])
        return
    def get_value(self):
        return self.obj.get_ydata()[0]
    def set_ydata(self,data):
        self.obj.set_ydata(data)
    def get_ydata(self):
        return self.obj.get_ydata()
    def set_value(self,val):
        x = self.obj.get_xdata()
        self.obj.set_ydata(np.ones(len(x))*val)


class LineScan(DraggableObj):
    roi_type = 'path'
    def __init__(self, obj, parent):
        DraggableObj.__init__(self, obj, parent)
        self.vconts = None
        ep_start,ep_stop = self.endpoints()
        ax = self.obj.axes
        self.length_tag = ax.text(ep_stop[0],ep_stop[1], '%2.2f'%self.length(),
                                  size = 'small',
                                  color = self.obj.get_color())
        self.name_tag = ax.text(ep_start[0],ep_start[1], self.obj.get_label(),
                                  size = 'small',
                                  color = self.obj.get_color())
        self.labels = [self.length_tag, self.name_tag]
        print("%2.2f"%self.length())
        self.update_tags()
        self.parent.legend()
        self.redraw()
    @property
    def has_traced_vessels(self):
        "Checks if LineScan instance has traced vessel data"
        return self.vconts is not None \
               and self.vconts.lcv is not None \
               and self.vconts.lcv.issteady
    def update_length_tag(self):
        "updates text with line length"
        lt = self.length_tag
        ep = self.check_endpoints()[1]
        lt.set_position(ep)
        lt.set_text('%2.2f'%self.length())
    def update_name_tag(self):
        "updates text with line length"
        nt = self.name_tag
        ep = self.check_endpoints()[0]
        nt.set_position(ep + (1,1))
        nt.set_rotation(self.get_slope())
    def get_slope(self):
        ep = self.check_endpoints()
        slope = np.arctan2(ep[1][1]-ep[0][1],ep[1][0]-ep[0][0])
        return np.rad2deg(-slope)
    def update_tags(self):
        self.update_name_tag()
        self.update_length_tag()

    def endpoints(self):
        return list(map(np.array, rezip(self.obj.get_data())))
    def centerpoint(self):
        x, y = self.obj.get_xdata(), self.obj.get_ydata()
        return [np.mean(x), np.mean(y)]
    def length(self):
        return coords.eu_dist(*self.endpoints())

    def check_endpoints(self):
        """Return endpoints in the order, such as
        the first point is the one closest to origin (0,0)"""
        pts = self.endpoints()
        iord = np.argsort(np.linalg.norm(pts,axis=1))
        return pts[iord[0]],pts[iord[1]]

    def shift(self, vec):
        ep = self.endpoints()
        vec = np.array(vec)
        self.obj.set_data((ep + vec).T)
        self.update_tags()

    def move(self, p):
        ep = self.endpoints()
        cp = np.array(self.centerpoint())
        p = np.array(p)
        xp,yp,k = self.pressed
        anchor = [cp, ep[0], ep[1]][k]
        d = p-anchor
        if k == 0: # closer to center point than to any end
            epnew = ep + d
            if self.parent.roi_layout_freeze:
                for roi in list(self.parent.roi_objs.values()):
                    if not roi == self:
                        roi.shift(d)
        elif k == 1:
            epnew = np.array([ep[0]+d, ep[1]])
        elif k == 2:
            epnew = np.array([ep[0], ep[1]+d])


        self.obj.set_data(epnew.T)
        self.update_tags()

    def on_press(self, event):
        if not self.event_ok(event, True):
            return
        if event.button == 3:
            self.destroy()
        elif event.button == 1:
            xy = event.xdata,event.ydata
            candidates = [self.centerpoint()] + self.endpoints()
            k = np.argmin([coords.eu_dist(xy, xp) for xp in candidates])
            self.pressed = event.xdata, event.ydata, k
        elif event.button == 2:
            self.dm = self.show_zview()

    def on_type(self, event):
        if not self.event_ok(event, True):
            return
        # press "/" for reslice, as in imagej
        accepted_keys = ['/', '-', 'r']
        if event.key in accepted_keys:
            self.diameter_manager = self.show_zview()

    def destroy(self):
        if self.tag in self.parent.roi_objs:
            self.parent.roi_objs.pop(self.tag)
        self.disconnect()
        for l in self.labels:
            l.set_alpha(0)
            l.remove()
        self.obj.remove()
        self.parent.legend()
        self.redraw()

    def transform_point(self, p):
        "convert Figure coordinates to pixels"
        dx,dy = self.axes[1:3]
        return p[0]/dx.value, p[1]/dy.value

    def get_zview_at_frames(self, frames, hwidth=2):
        points = list(map(self.transform_point, self.check_endpoints()))
        dx,dy = [_x.value for _x in self.axes[1:3]]
        #((dx,xunits), (dy,yunits)) = map(quantity_to_pair, self.axes[1:3])
        #print 'points:', points[0][0], points[0][1],
        #print 'points:', points[0][0]*dx, points[0][1]*dy
        #print 'points type is:', map(type, points)
        tv = np.reshape(translate(*points),-1) # translation vector
        #timeview = lambda pts: np.array([line_reslice3(frame, *pts) for frame in frames()])
        timeview = lambda pts: np.array([line_reslice3(frame, *pts) for frame in frames])

        if hwidth > 0:
            plist = [points]
            plist += [points + s*k*tv for k in range(1, hwidth+1) for s in (-1,1)]
            out = np.mean(list(map(timeview, plist)), axis=0)
        else:
            out = timeview(points)
        return np.squeeze(np.rot90(out)),points


    def get_zview(self,hwidth=2,frange=None):
        if frange is None:
            if hasattr(self.parent, 'caller'): # is called from frame_viewer?
                caller = self.parent.caller
                hwidth = caller.fso.linescan_width # overrides argument
                half_scope = caller.fso.linescan_scope
                if half_scope <= 0:
                    frames = self.parent.active_stack
                else:
                    fi = caller.frame_index
                    fstart = max(0,fi-half_scope)
                    fstop = min(len(self.parent.active_stack),fi+half_scope)
                    frange = slice(fstart,fstop)
                    frames = self.parent.active_stack[frange]
            else:
                frames = self.parent.active_stack
        else:
            frames = self.parent.active_stack[frange]
        return self.get_zview_at_frames(frames, hwidth=hwidth)

    def _get_full_projection(self, fn = np.mean,axis=1,
                            mode='constant',
                            output = 'result'):
        """
        if "output" is 'function' return a function to rotate and project any  data
        if "output is 'result', return a projection of rotated data, associated
        with the frame sequence of the Picker.
        """
        from scipy.ndimage.interpolation import rotate
        points = list(map(self.transform_point, self.check_endpoints()))
        k, b = line_from_points(*points)
        phi = np.rad2deg(np.arctan(k))
        def _(data):
            rot = rotate(data, phi, (1,2), mode=mode)
            rot = np.ma.masked_less_equal(rot, 0)
            return np.array(fn(rot, axis=axis))
        if output=='result':
            return _(self.parent.frame_coll[:])
        elif output == 'function':
            return _

    def show_zview(self,frange=None,hwidth=2):
        timeview,points = self.get_zview(frange=frange,hwidth=hwidth)
        gw_meas = [None]


        if timeview is not None:
            data = [timeview]
            frame_coll = self.parent.frame_coll
            fig, ax = plt.subplots()
            plt.subplots_adjust(bottom=0.2)

            # TODO: work out if sx differs from sy

            #(dz,zunits), (dx,xunits) = frame_coll.meta['axes'][:2]
            dz,dx,dy = frame_coll.meta['axes']
            x_extent = (0,dz.value*timeview.shape[1])
            if mpl.rcParams['image.origin'] == 'lower':
               lowp = 1
            else:
               lowp = -1
            ax.imshow(timeview,
                      extent=x_extent + (0, dx.value*timeview.shape[0]),
                      interpolation='nearest',
                      aspect='auto')

            #xlabel = hasattr(dz, 'unit') and dx.unit or 'frames'
            xlabel = dx.unit if hasattr(dx,'unit') else 'frames'
            ylabel = dz.unit if hasattr(dz,'unit') else 'pixels'
            #xlabel = dz.unit.is_dimensionless and 'frames' or dz.unit
            #ylabel = dx.unit.is_dimensionless and 'pixels' or dx.unit


            ax.set_ylabel(ylabel)
            ax.set_xlabel(xlabel)
            ax.set_title('Timeview for '+ self.tag)

            def _inv_callback(event):
                data[0] = -data[0]
                plt.setp(ax.images[0],
                         data=data[0],
                         clim=(data[0].min(),data[0].max()))
                plt.show() #this should solve the issue with non-activated
                           #VesselTracker window

            def _vessel_callback(event):
                print('Vessel wall tracking')
                self.vconts = VesselContours(data[0],self.tag)
                plt.show() #this should solve the issue with non-activated
                           #VesselTracker window

            self.buttons = []

            ax_diam = plt.axes([0.1, 0.025, 0.2, 0.05])
            btn_diam = mw.Button(ax_diam, "Trace vessel")
            btn_diam.on_clicked(_vessel_callback)
            self.buttons.append(btn_diam)

            ax_inv = plt.axes([0.32, 0.025, 0.2, 0.05])
            btn_inv = mw.Button(ax_inv, "Invert data")
            btn_inv.on_clicked(_inv_callback)
            self.buttons.append(btn_inv)
            plt.show()

        return btn_diam

    @property
    def color(self):
        return self.obj.get_color()


    def to_struct(self):
        l = self.obj
        return {'func': line_from_struct,
                'xdata': l.get_xdata(),
                'ydata': l.get_ydata(),
                'alpha': l.get_alpha(),
                'label': l.get_label(),
                'color': l.get_color(),}


class CircleROI(DraggableObj):
    """Draggable Circle ROI"""
    step = 1
    roi_type = 'area'
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
            print(event.key)
        tags = [self.tag]
        if event.key in ['t', '1']:
            self.parent.show_zview(tags)
        elif event.key in ['c']:
            self.parent.show_xcorrmap(self.tag)
        elif event.key in ['T', '!']:
            self.parent.show_zview(tags, normp=True)
        elif event.key in ['w', '2']:
            self.parent.show_spectrogram_with_ts(self.tag)
        elif event.key in ['W', '3']:
            self.parent.show_wmps(tags)
        elif event.key in ['4']:
            self.parent.show_ffts(tags)

    def on_press(self, event):
        if not self.event_ok(event, True): return
        x0,y0 = self.obj.center
        if event.button == 1:
            self.pressed = event.xdata, event.ydata, x0, y0
        elif event.button == 2:
            self.parent.show_zview([self.tag])
        elif event.button == 3:
            self.destroy()

    def destroy(self):
        p = self.parent.roi_objs.pop(self.tag)
        self.disconnect()
        self.obj.remove()
        self.tagtext.remove()
        self.redraw()

    def shift(self, vec):
        self.obj.center += np.array(vec)
        self.set_tagtext()
        self.redraw()

    def move(self, p):
        "Move the ROI when the mouse is pressed over it"
        xp,yp, x, y = self.pressed
        dx = p[0] - xp
        dy = p[1] - yp
        new_center = np.array((x+dx, y+dy))
        shift =  new_center - self.obj.center
        self.obj.center = new_center

        if self.parent.roi_layout_freeze:
            # drag all other ROIs along
            for roi in list(self.parent.roi_objs.values()):
                if not roi == self:
                    roi.shift(shift)
        self.set_tagtext()
        self.redraw()

    center = property(lambda self: self.obj.center,
                      move)

    def set_tagtext(self, margin=2, angle=np.pi/2):
        pi = np.pi
        ax = self.obj.axes
        p = self.obj.center
        r = self.obj.get_radius() + margin
        x = p[0] + np.cos(angle)*r
        y = p[1] + np.sin(angle)*r
        lowp = mpl.rcParams['image.origin'] == 'lower'
        if lowp : angle = -angle
        if -pi/4 <= angle < pi/4:
            ha, va = 'left', 'center'
        elif -3*pi/4 <= angle < -pi/4:
            ha,va = 'center', 'bottom'
        elif pi/4 <= angle < 3*pi/4 :
            ha,va = 'center', 'top'
        else:
            ha,va = 'right', 'center'
        #print 'ha: {}, va: {}'.format(ha, va)
        if not hasattr(self, 'tagtext'):
            self.tagtext = ax.text(x,y, '{}'.format(self.obj.get_label()),
                                   verticalalignment=va,
                                   horizontalalignment=ha,
                                   size = 'small',
                                   color = self.color)
        else:
            self.tagtext.set_position((x,y))

    def in_circle(self, shape):
        roi = self.obj

        dx,dy = [x.value for x in self.axes[1:3]]
        c = roi.center[0]/dx, roi.center[1]/dy
        fn = coords.in_circle(c, roi.radius/dx)
        X,Y = np.meshgrid(*list(map(range, shape[::-1])))
        return fn(X,Y)

    def get_zview(self, normp=False):
        """return z-series (for timelapse --- timeseries) from roi

        Parameters:
          - if normp is False, returns raw timeseries v
          - if normp is a function f, returns f(v)
          - if normp is a number N, returns :math:`\\Delta v/v_0`, where
            :math:`v_0` is calculated over N first points
          - else, returns :math:`\\Delta v/\\bar v`

        """

        fullshape = self.parent.active_stack.frame_shape
        sh = fullshape[:2]
        v = self.parent.active_stack.mask_reduce(self.in_circle(sh))
        #print len(fullshape), hasattr(self.parent.fseq, 'ch'), self.parent.fseq.ch
        if len(fullshape)>2 and hasattr(self.parent.active_stack, 'ch') \
           and (self.parent.active_stack.ch is not None):
            v = v[:,self.parent.active_stack.ch]
        if normp:
            if isinstance(normp, collections.Callable):
                return normp(v)
            else:
                Lnorm = isinstance(normp,int) and normp or len(v)
                return DFoF(v, Lnorm)
        else: return v

    def to_struct(self):
        c = self.obj
        return {'func' : circle_from_struct,
                'center': c.center,
                'radius': c.radius,
                'alpha': c.get_alpha(),
                'label': c.get_label(),
                'facecolor':c.get_facecolor(),
                'linewidth':c.get_linewidth(),
                'edgecolor': c.get_edgecolor(),}


class DragRect(DraggableObj):
    "Draggable Rectangular ROI"
    roi_type = 'area'
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
    return 0.5 + np.sum(arr1*arr2)/(2*arr1.std()*arr2.std())


class RectFollower(DragRect):
    def __init__(self, obj, *args, **kwargs):
        DragRect.__init__(self, obj, *args, **kwargs)
        self.search_width = 1.5*obj.get_width()
        self.search_height = obj.get_width()
    def on_type(self, event):
        if not self.event_ok(event, True): return
    def xy(self):
        return list(map(int, self.obj.get_xy()))
    def toslice(self,xoff=0,yoff=0):
        start2,start1 = self.xy()
        start2 += xoff
        start1 += yoff
        w, h = self.obj.get_width(), self.obj.get_height()
        return (slice(start1, start1+h), slice(start2, start2+w))
    def find_best_pos(self, frame, template, measure=sse_measure):
        acc = {}
        sw, sh = self.search_width, self.search_height
        limh, limw = frame.shape
        for w in np.arange(max(0,-sw/2), min(sw/2,limw)):
            for h in np.arange(max(0,-sh/2), min(limh,sh/2)):
                s = self.toslice(h,w)
                d = measure(core.rescale(frame[s]), template)
                acc[(w,h)] = d
        pos = sorted(list(acc.items()), lambda x, y: cmp(x[1], y[1]), reverse=True)
        pos = pos[0][0]
        x,y = self.xy()
        return x+pos[0], y+pos[1]



def synthetic_vessel(nframes, width = 80, shape=(512,512), noise = 0.5):
    left = 100
    right = left+width
    frames = []
    xw1 = core.ar1()
    xw2 = core.ar1()
    for i in range(nframes):
        f = np.zeros(shape)
        l,r= 2*left+next(xw1),2*right+next(xw2)
        f[:,l:l+4] = 1.0
        f[:,r:r+4] = 1.0
        f += np.random.randn(*shape)*noise
        frames.append(((f.max()-f)/f.max())*255.0)
    return np.array(frames)

class VesselContours(object):
    """
    UI for finding vessel contours in cross-lines
    """
    def __init__(self,data, title=''):
        self.upsampled=1
        self.data = data
        self.lcv = None
        f, axs = plt.subplots(3,1,sharex=True)
        self.fig = f
        self.axs = axs
        th = 0.5

        axs[0].imshow(data, aspect='auto', interpolation='nearest',
                      cmap='gray')
        a = axs[0].axis()

        self.start_seed = self.set_auto_seeds()
        self.contlines = None
        self.startlines = [axs[0].plot(s,'g')[0]
                           for s in self.start_seed]
        self.startlines2 = [ThresholdLine(l,self) for l in self.startlines]

        self.snrv = core.misc.simple_snr2(data)

        axs[1].plot(self.snrv, 'k',ls='steps')
        lh = axs[1].axhline(th, color='blue', ls='-')
        self.th_line = ThresholdLine(lh,self)

        d = self.start_seed[1]-self.start_seed[0]
        self.diam_line = axs[2].plot(d,'k--')[0]
        axs[2].set_ylim(0, d[0]*1.2)

        axs[0].axis(a)
        self.config_axes()
        axs[0].set_title(title)
        self.set_buttons()
        return

    def config_axes(self):
        self.fig.set_facecolor('white')
        corners1 = [0.1, 1-0.4, 0.8,0.35]
        corners2 = [0.1, 0.5-0.05, 0.8, 0.1]
        corners3 = [0.1, 0.15, 0.8, 0.25]

        for a, c in zip(self.axs, [corners1, corners2, corners3]):
            a.set_position(c)

        plt.setp(self.axs[1], yticks=[], frame_on=False,
                 ylabel='rel. SNR')
        plt.setp(self.axs[2], ylabel="diameter, px",
                 xlabel='frame #',
                 frame_on=False)
        return

    def set_buttons(self):
        f = self.fig
        axstart = f.add_axes([0.1, 0.02, 0.2, 0.05])
        self.axref = f.add_axes([0.31, 0.02, 0.2, 0.05])
        self.btn_start = mw.Button(axstart,'Start')
        self.btn_start.on_clicked(self.solve_contours)
        self.btn_ref = mw.Button(self.axref,'Refine')
        self.btn_ref.on_clicked(self.refine_solution)
        plt.setp(self.axref, visible=False)
        f.canvas.draw()
        return

    def refine_solution(self, event,upsample=3):
        if not self.axref.get_visible():
            return
        data = core.misc.ainterpolate(self.data,n=upsample)

        start = self.lcv.conts.reshape(2,-1)
        if  self.upsampled < upsample:
            start *= upsample
        self.solve_contours(None,start,data,upsample)

    def solve_contours(self,event,start=None,data = None,
                       upsample=1.,
                       nmax=500,skipdraw=3):
        self.upsampled=upsample
        if data is None:
            data = self.data
        if start is None:
            start = [l.get_ydata() for l in
                     self.startlines2]

        for k in range(2):
            self.startlines2[k].set_ydata(start[k]/upsample)
        if self.contlines is None:
            self.contlines = [self.axs[0].plot(s,'orange')[0]
                              for s in self.start_seed]
        else:
            for lhfrom,lhto in zip(self.startlines2, self.contlines):
                lhto.set_ydata(lhfrom.get_ydata())

        th = self.th_line.get_value()
        lcv = track.LCV_Contours(start, data, thresh=th)
        for i in range(nmax):
            lh = lcv.verlet_step()
            for c,v in zip(self.contlines,lh):
                c.set_ydata(v/upsample)
            d = lcv.diameter/upsample
            self.diam_line.set_ydata(d)
            if not i%skipdraw:
                self.fig.canvas.draw()
            if lcv.issteady:
                for c in self.contlines:
                    plt.setp(c, lw=1,color='r')
                self.diam_line.set_linestyle('-')
                self.axs[2].set_ylim(d.min()*0.9, d.max()*1.1)
                plt.setp(self.axref,visible=True)
                break
        self.lcv = lcv
        self.fig.canvas.draw()

    @property
    def diameter(self):
        if self.lcv is None: return
        d = self.lcv.diameter
        return d/self.upsampled

    def set_auto_seeds(self):
        d = self.data
        ext = d.shape[0]
        margin = ext*0.1
        try:
            seeds = track.guess_seeds(d.T,-1)
            if np.abs(seeds[1]-seeds[0]) < margin:
                locw,highc = 2*margin, ext-2*margin
        except Exception as e:
            print("couldn't automatically set starting seeds:", e)
            seeds = ext*0.1+margin, ext*0.9-margin
        lowc = np.ones(d.shape[1])*seeds[0] - margin
        lowc = np.where(lowc < 0, margin, lowc)
        highc = np.ones(d.shape[1])*seeds[1] + margin
        highc = np.where(highc>ext,  ext-margin, highc)
        self.auto_seeds =  (lowc, highc)
        return (lowc, highc)
