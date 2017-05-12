 # a/b will always return float

import itertools as itt
import os
from functools import partial

import numpy as np

import scipy.io
from scipy import ndimage

import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import widgets as mw
from matplotlib import path



import pickle
import random
import collections
from functools import reduce

try:
    from swan import pycwt, utils
    from swan.gui import swanrgb
except:
    "Can't load swan (not installed?)"

_wavelet_cmap = plt.cm.viridis

try:
    import pandas as pd
    _with_pandas = True
except:
    "Can't load pandas (not installed?)"
    _with_pandas = False


from .. import fnmap
from .. import track
from .. import core
from .. import io
from ..core import ifnot,rezip
from ..core.units import quantity_to_pair
from ..core.baselines import DFoSD, DFoF
from .plots import group_maps, group_plots
from . import DraggableRois
from .DraggableRois import CircleROI, LineScan

_widgetcolor = 'lightyellow'

_picker_help_msg="""
Picker
=======
Simple interactive gui to operate frame sequence using matplotlib
for vis, interaction and widgets.

Usage:
------

Mouse:
~~~~~~
  - Left-click to create a Circle ROI (squares to be added)
  - Right-click-and-drag to create a Line ROI
  - Right-click on any ROI to remove it
  - Middle-click on a ROI to get z-slice for Line ROIs and time series for
    Circle ROIs
  - Scroll while hovering mouse over any ROI to change its size
  - Scroll while hovering mouse outside any ROI to go through frames

Keyboard:
~~~~~~~~~
  - press `h` to show 'home' frame
  - press Shift to enable Lasso selector
  - press `t` or `1` with mouse over Circle ROI to show corresponding time series
  - press `w` or `2` over Circle ROI to show wavelet spectrogram
  - press `3` over Circle ROI to show time-averaged wavelet power spectrum
  - press `4` over Circle ROI to show FFT power spectrum
  - press `/` or '-' or 'r' with mouse over Line ROI to get z-reslice
"""

from imfun import fseq

def color_walker():
    ar1 = core.ar1
    red, green, blue = ar1(), ar1(), ar1()
    while True:
        yield [np.mod(next(x),1.0) for x in (red,green,blue)]


def rand_tag():
    vowels = "aeiouy"
    consonants = "qwrtpsdfghjklzxcvbnm"
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
        tag = next(tagger)
        n += 1
        if not tag in tags:
            return tag
    return "Err"


_default_roi_coloring = 'allrandom' # {allrandom | groupsame | groupvar}

class Picker (object):
    _verbose = False
    def __init__(self, frames, home_frame = True, verbose=False,
                 roi_coloring_model=_default_roi_coloring, 
                 suptitle = None,
                 roi_prefix = 'r',
                 default_circle_rad = 5.,
                 min_linescan_length = 1.):
        self._corrfn = 'pearson'
        self.cw = color_walker()
        self.suptitle = suptitle
        self._show_legend=False
        if isinstance(frames, fseq.FStackColl):
            frame_coll = frames
        elif isinstance(frames, fseq.FrameStackMono):
            frame_coll = fseq.FStackColl([frames])
        else:
            print("Unrecognized frame stack format. Must be either derived from fseq.FrameStackMono or fseq.FStackColl")
            return
        self.frame_coll = frame_coll
        self._Nf = None
        self.roi_coloring_model = roi_coloring_model
        self.roi_objs = {}
        self._tag_pallette = {}
        self.roi_prefix = roi_prefix
        self.current_color = next(self.cw)
        self.default_circle_rad = default_circle_rad
        self.min_length = min_linescan_length
        self.frame_index = 0
        self.shift_on = False
        self.roi_layout_freeze = False
        #self.widgetcolor = 'lightyellow'
        self.frame_slider = None
        self._verbose=verbose

        self._init_ccmap()
        f = self._init_home_frame(home_frame)

        ## set lut color limits
        # -- omitting  vmin,vmax, or clim arguments for now
        splitted = np.split(f, frame_coll.nCh,-1)
        clims = [(np.min(_x),np.max(_x)) for _x in splitted]
        self.clims = clims
        self.home_frame = f
        self.frame_cache = {}
        self._use_cache_flag = True
        self.cmap='gray'
        return

    def _init_home_frame(self, home_frame):
        ## set home_frame
        dtype = self.frame_coll[0].dtype
        if isinstance(home_frame,  np.ndarray):
            f = home_frame
        elif isinstance(home_frame, collections.Callable):
            f = self.frame_coll.time_project(home_frame)
            f = f.astype(dtype)
            pass
        elif home_frame == 'mean' or (isinstance(home_frame, bool) and home_frame):
            f = self.frame_coll.mean_frame().astype(dtype)
        else:
            f = self.frame_coll[0]

        return f

    def _init_ccmap(self):
        self._ccmap = {key:None for key in 'irgb'}
        nCh = self.frame_coll.nCh
        if nCh == 1:
            self._ccmap['i'] = 0
        else:
            for k,c in zip(list(range(nCh)),'rgb'):
                self._ccmap[c] = k

    def _lut_controls(self,event=None):
        """
        Simple gui to control a mapping between RGB color display channels and
        data streams/channels
        """
        fig = plt.figure(figsize=(1,5))

        channels = 'rgbi'
        channel_names = ['Red','Green','Blue','Grayscale']
        streams = [s.meta['channel'] for s in self.frame_coll.stacks]
        stream_idx = {s:k for k,s in enumerate(streams)}

        nCh = self.frame_coll.nCh
        self.channel_ctrls = {k:None for k in channels}


        spacing =0.05
        el_h = (1-(1+len(channels))*spacing)/len(channels)

        def _update_choices(event):
            for key,val in list(self.channel_ctrls.items()):
                selected = val.value_selected
                if selected == '--' or selected is None:
                    ch = None
                else:
                    ch = stream_idx[selected]
                    if key is 'i':
                        self.ach_setter.set_active(ch)
                self._ccmap[key] = ch
            self.frame_cache = {}
            pass

        for k,c in enumerate(channels):
            ax = fig.add_axes([0.1,  1 - (k+1)*(el_h +spacing), 0.8, el_h ], aspect='equal')
            ax.set_title(channel_names[k],size='small',color= (c!='i') and c or 'k')
            active = (k+1)%4
            if k > nCh: active = 0
            channel_selector = mw.RadioButtons(ax, ['--'] + streams, active = active)
            channel_selector.on_clicked(_update_choices)
            self.channel_ctrls[c] = channel_selector
        plt.subplots_adjust(left=0.05,right=0.95,bottom=0.05,top=0.95)

    def _levels_controls(self,event=None):
        fig = plt.figure(figsize=(5,3))
        spacing = 0.03
        wspacing = 0.1
        nCh = self.frame_coll.nCh
        #el_h = 0.05
        el_h = (1 - (1+2*nCh)*spacing)/(2*nCh)
        el_w = 0.5
        self.level_controls = {}
        streams = [s.meta['channel'] for s in self.frame_coll.stacks]
        stream_idx = {s:k for k,s in enumerate(streams)}
        def _update_levels(event):
            for k in range(nCh):
                low,high = self.level_controls[k]
                self.clims[k] = (low.val, high.val)
            self.frame_cache = {}
            pass
        for k,stack in enumerate(self.frame_coll.stacks):
            channel_name = stack.meta['channel']
            ax_high = fig.add_axes([0.2, 1- (2*k+1)*(el_h+spacing)-spacing, el_w, el_h],
                                   aspect='auto', facecolor=_widgetcolor)
            ax_low = fig.add_axes([0.2, 1- (2*k+2)*(el_h+spacing), el_w, el_h],
                                  aspect='auto', facecolor=_widgetcolor)

            lmin,lmax = stack.data_range()
            low,high = self.clims[k]
            low_slider = mw.Slider(ax_low, '%s low'%channel_name, lmin,lmax,valinit=low)
            high_slider = mw.Slider(ax_high, '%s high'%channel_name, lmin,lmax,valinit=high)

            low_slider.on_changed(_update_levels)
            high_slider.on_changed(_update_levels)
            self.level_controls[k] = (low_slider,high_slider)
        plt.subplots_adjust(left=0.05,right=0.95,bottom=0.05,top=0.95)


    def _init_active_channel_setter(self):
        """
        Sets which color channel is active for ROI interaction
        """
        corners = self.ax1.get_position().get_points()
        height = 0.15
        width = 0.1
        left = 0.03
        bottom = corners[1,1]-height

        ax_active = self.fig.add_axes([left,bottom,width,height],
                                      aspect='equal',facecolor=_widgetcolor)

        ax_active.set_title('Active channel',size='small')

        streams = [s.meta['channel'] for s in self.frame_coll.stacks]
        stream_idx = {s:k for k,s in enumerate(streams)}

        self.active_stack = self.frame_coll.stacks[0]
        self.ach_setter = mw.RadioButtons(ax_active, streams)
        def _update_active(event):
            ind = stream_idx[self.ach_setter.value_selected]
            #print 'active stack updated to ', ind
            self.active_stack = self.frame_coll.stacks[ind]

        self.ach_setter.on_clicked(_update_active)
        pass

    def start(self, roi_objs={}, ax=None, legend_type = 'figlegend',
              **imshow_args):
        "Start picking up ROIs"
        self.tagger = tags_iter()
        #self.drcs = {}
        self.frame_hooks = {}
        self.frame_slider = None
        Nf = len(self.frame_coll)

        button_hmargin = 0.03
        button_vmargin = 0.01
        button_width = 0.1
        button_height = 0.05
        self.ui_buttons = {}

        if ax is None:

            self.fig, self.ax1 = plt.subplots()
            if self.suptitle:
                self.fig.suptitle(self.suptitle)
            plt.subplots_adjust(left=0.2, bottom=0.2)
            corners = self.ax1.get_position().get_points()
            #print corners
            axfslider = plt.axes([corners[0,0], 0.1, corners[1,0]-corners[0,0], 0.03], facecolor=_widgetcolor)
            self.frame_slider = mw.Slider(axfslider, 'Frame', 0, Nf-1, valinit=0,
                                          valfmt='%d')
            self.frame_slider.on_changed(self.set_frame_index)
            self._init_active_channel_setter()

            yloc = corners[1,1] - 3*(2*button_vmargin+button_height)
            bax_lut = self.fig.add_axes([button_hmargin,
                                         yloc,
                                         button_width, button_height],)
            lut_but = mw.Button(bax_lut, 'Colors', color=_widgetcolor)
            lut_but.on_clicked(self._lut_controls)
            self.ui_buttons['lut'] = lut_but

            yloc -= (button_vmargin+button_height)
            bax_lev = self.fig.add_axes([button_hmargin, yloc, button_width, button_height])
            lev_but = mw.Button(bax_lev, 'Levels', color=_widgetcolor)
            lev_but.on_clicked(self._levels_controls)
            self.ui_buttons['levels'] = lev_but

            yloc -= (button_vmargin+button_height)
            bax_freeze = self.fig.add_axes([button_hmargin,  yloc, button_width, button_height])
            freeze_but = mw.Button(bax_freeze, 'Freeze ROIs',color=_widgetcolor)
            def _freeze_control(event):
                if self.roi_layout_freeze:
                    freeze_but.color = _widgetcolor
                    self.roi_layout_freeze = False
                else:
                    freeze_but.color = 'pink'
                    self.roi_layout_freeze = True
            freeze_but.on_clicked(_freeze_control)
            self.ui_buttons['freeze'] = freeze_but

            yloc -= (button_vmargin + button_height)
            bax_drop = self.fig.add_axes([button_hmargin,  yloc, button_width, button_height])
            drop_but = mw.Button(bax_drop, 'Drop ROIs',color=_widgetcolor)
            drop_but.on_clicked(self.drop_all_rois)
            self.ui_buttons['drop'] = drop_but

            for button in list(self.ui_buttons.values()):
                button.label.set_fontsize('small')
        else:
            self.ax1 = ax
            self.fig = self.ax1.figure
        self.legtype = legend_type
        self.pressed = None

        ## setup axes and show home frame
        axes = self.frame_coll.meta['axes']

        dy,dx = axes[1:3]
        yunits, xunits = (str(x.unit) for x in (dy,dx))
        sy,sx = self.frame_coll[0].shape[:2]

        iorigin = mpl.rcParams['image.origin']
        lowp = [1,-1][iorigin == 'upper']

        if 'cmap' in imshow_args:
            self.cmap = imshow_args.pop('cmap')
        else:
            self.cmap='gray'

        self.plh = self.ax1.imshow(self._lutconv(self.home_frame),
                                   extent = (0, sx*dx.value)+(0, sy*dy.value)[::lowp],
                                   aspect='equal',
                                   cmap= self.cmap,
                                   **imshow_args)
        self.ax1.set_xlabel(str(yunits))
        self.ax1.set_ylabel(str(xunits))

        self.disconnect()
        self.connect()
        self.fig.canvas.draw()
        if self._verbose:
            print(_picker_help_msg)
        return self.ax1, self.plh, self


    def length(self):
        if self._Nf is None:
            self._Nf = len(self.frame_coll)
        return self._Nf


    def legend(self):
        if self._show_legend == True:
            keys = sorted(self.roi_objs.keys())
            handles = [self.roi_objs[key].obj for key in keys]
            try:
                axs= self.ax1.axis
                if self.legtype is 'figlegend':
                    plt.figlegend(handles, keys, 'upper right')
                elif self.legtype is 'axlegend':
                    self.ax1.legend(handles, keys)
                    self.ax1.axis(axs)
                    self.redraw()
            except Exception as e:
                    print("Picker: can't make legend because ", e)

    def event_canvas_ok(self, event):
        "check if event is correct axes and toolbar is not in use"
        pred = event.inaxes !=self.ax1 or \
                   self.any_roi_contains(event) or \
                   self.fig.canvas.toolbar.mode !=''
        return not pred


    def on_modkey(self, event):
        if not self.event_canvas_ok(event):
            return
        if event.key == 'shift':
            if not self.shift_on:
                self.shift_on = True
                for spine in list(self.ax1.spines.values()):
                    spine.set_color('r')
                    spine.set_linewidth(4)
            else:
                self.shift_on = False
                for spine in list(self.ax1.spines.values()):
                    spine.set_color('k')
                    spine.set_linewidth(1)
        self.ax1.figure.canvas.draw()
        return

    def on_keyrelease(self, event):
        return

    def init_line_handle(self):
        color = self.new_color()
        lh, = self.ax1.plot([0],[0],'-', color=color)
        return lh

    def new_roi_tag(self):
        "make new tag/label for a ROI"
        pref = self.roi_prefix
        matching_tags = [t for t in self.roi_tags() if t.startswith(pref)]
        for n in range(1,int(1e5)):
            newtag = pref + '{:02d}'.format(n)
            if newtag not in matching_tags:
                break
        return newtag

    def new_color(self):
        if self.roi_coloring_model == 'allrandom':
            self.current_color = next(self.cw)
        elif self.roi_coloring_model == 'groupsame':
            self.current_color = self._tag_pallette[self.roi_prefix]
        elif self.roi_coloring_model == 'groupvar':
            base_color = self._tag_pallette[self.roi_prefix]
            var_color = np.array(next(self.cw))*0.1
            self.current_color = list(np.clip(base_color+var_color, 0,1))
        return self.current_color

    @property
    def roi_prefix(self):
        "roi prefix property"
        return self._roi_prefix_x
    @roi_prefix.setter
    def roi_prefix(self, pref):
        matching_rois = self.roi_tags(lambda x: x.startswith(pref))
        if not len(matching_rois):
            self._tag_pallette[pref] = next(self.cw)
        self.current_color = self._tag_pallette[pref]
        self._roi_prefix_x = pref


    def on_click(self,event):
        if not self.event_canvas_ok(event): return
        #print('in on click')
        #x,y = round(event.xdata), round(event.ydata)
        x,y = event.xdata, event.ydata # do I really need to round?
        #print('button is', event.button, event.button == 1)
        #print (self.any_roi_contains(event), 
        #       self.shift_on, self.roi_layout_freeze)
        if event.button == 1 and \
           not self.any_roi_contains(event) and \
           not self.shift_on and \
           not self.roi_layout_freeze:
            #print("In if statement!")
            #label = unique_tag(self.roi_tags(), tagger=self.tagger)
            label = self.new_roi_tag()
            #print("new label", label)
            dx,dy = self.frame_coll.meta['axes'][1:3]

            #color = self.cw.next()
            color = self.new_color()
            c = plt.Circle((x,y), self.default_circle_rad*dx.value,
                           label = label,
                           linewidth = 1.5,
                           facecolor=color+[0.5],
                           edgecolor=color)

            c.figure = self.fig
            self.ax1.add_patch(c)
            self.roi_objs[label]= CircleROI(c, self)
        elif event.button == 3 and \
             not self.shift_on and \
             not self.roi_layout_freeze:
            self.pressed = event.xdata, event.ydata
            axrange = self.ax1.axis()
            self.curr_line_handle = self.init_line_handle()
            self.ax1.axis(axrange)
        elif self.shift_on:
            if not self.ax1.figure.canvas.widgetlock.locked():
                #self.lasso = mw.Lasso(event.inaxes, (x, y), self.lasso_callback)
                self.lasso = mw.LassoSelector(event.inaxes,
                                              self.lasso_callback,
                                              lineprops=dict(color='g',lw=2))
                self.ax1.figure.canvas.widgetlock(self.lasso)
        self.legend()
        self.ax1.figure.canvas.draw()

    def lasso_callback(self, verts):
        "from Lasso widget, get a mask, which is 1 inside the Lasso"
        # TODO! update to usage of FStackColl
        p = path.Path(verts)
        sh = self.active_stack.frame_shape
        print('Active stack shape', sh)
        locs = list(itt.product(*map(range, sh[::-1])))
        dy,dx = self.active_stack.meta['axes'][1:3]
        out = np.zeros(sh)
        xys = np.array(locs, 'float')
        xys[:,0] *= dy.value
        xys[:,1] *= dx.value # coordinates with scale
        ind = p.contains_points(xys)
        for loc,i in zip(locs, ind):
            if i : out[loc[::-1]] = 1
        self.ax1.figure.canvas.draw()
        self.ax1.figure.canvas.widgetlock.release(self.lasso)
        del self.lasso
        f = plt.figure()
        ax = f.add_subplot(111)
        vmin,vmax = self.plh.get_clim()
        print(vmin, vmax)
        print(out.shape)
        #print self.home_frame.shape
        #ax.imshow(self.home_frame, cmap='gray',vmin=vmin,vmax=vmax)
        hf = self._lutconv(self.home_frame)
        print(hf.shape)
        ax.imshow(hf, cmap=self.cmap)
        ax.contour(out, levels=[0],colors=['g'])
        self.pmask = out
        return

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
            #tag = unique_tag(self.roi_tags(), tagger=self.tagger)
            tag = self.new_roi_tag()
            self.curr_line_handle.set_label(tag)
            newline = LineScan(self.curr_line_handle, self)
            if newline.length() > self.min_length:
                self.roi_objs[tag] = newline
            else:
                try:
                    newline.destroy()
                    self.curr_line_handle.remove()
                except Exception as e:
                    #print "Can't remove line handle because", e
                    pass
        else:
            try:
                self.curr_line_handle.remove()
            except Exception as e:
                print("Can't remove line handle because", e)
        self.curr_line_handle = self.init_line_handle()
        self.legend()
        self.fig.canvas.draw() #todo BLIT!
        return


    def any_roi_contains(self,event):
        "Checks if event is contained in any ROI"
        if len(self.roi_objs) < 1 : return False
        return reduce(lambda x,y: x or y,
                      [roi.obj.contains(event)[0]
                       for roi in list(self.roi_objs.values())])

    def roi_tags(self, filt = lambda x:True):
        "List of tags for all ROIs"
        return sorted(list(filter(filt, self.roi_objs.keys())))

    def export_rois(self, fname=None):
        """Exports picked ROIs as structures to a file and/or returns as list"""
        out = [x.to_struct() for x in list(self.roi_objs.values())]
        if fname:
            pickle.dump(out,
                        open(fname, 'wb'), protocol=0)
            if self._verbose:
                print("Saved ROIs to ", fname)
        return out

    def load_rois(self, source):
        "Load stored ROIs from a file"
        if isinstance(source, str):
            data = pickle.load(open(source,'rb'))
        elif isinstance(source, file):
            data = pickle.load(source)
        else:
            data = source
        rois = [x['func'](x) for x in data]
        #circles = [x for x in rois if isinstance(x, plt.Circle)]
        #lines = [x for x in rois if isinstance(x, plt.Line2D)]
        for x in rois:
            if isinstance(x,plt.Circle):
                self.ax1.add_patch(x)
                constructor = CircleROI
            if isinstance(x,plt.Line2D):
                self.ax1.add_line(x)
                constructor = LineScan
            self.roi_objs.update(dict([(x.get_label(),constructor(x,self))]))                
        #list(map(self.ax1.add_patch, circles)) # add points to the axes
        #list(map(self.ax1.add_line, lines)) # add points to the axes
        #self.roi_objs.update(dict([(c.get_label(), CircleROI(c,self))
        #                           for c in circles]))
        #self.roi_objs.update(dict([(l.get_label(), LineScan(l,self))
        #                           for l in lines]))

        #self.ax1.legend()
        self.ax1.figure.canvas.draw() # redraw the axes
        return
    def drop_all_rois(self,event):
        for roi in list(self.roi_objs.values()):
            roi.destroy()

    def trace_vessel_contours_in_all_linescans(self, hwidth=2):
        for tag in self.roi_objs:
            if self.isLineROI((tag)):
                roi = self.roi_objs[tag]
                data,_ = roi.get_zview(hwidth=hwidth)
                roi.vconts = DraggableRois.VesselContours(data, tag)
        plt.show()

    def export_vessel_diameters(self,fname=None,save_figs_to=None, format='csv'):
        objs = self.roi_objs
        keys = [k for k in sorted(objs.keys())
                if self.isLineROI(k) and objs[k].has_traced_vessels]
        if not len(keys):
            if self._verbose:
                print("No LineScane ROIs with traced vesels found")
            return
        if save_figs_to is not None:
            for k in keys:
                vcont_obj = objs[k].vconts
                fig,ax = plt.subplots(1,1)
                ax.imshow(vcont_obj.data, cmap=self.cmap)
                ax_lim = ax.axis()
                ax.plot(vcont_obj.contlines[0].get_ydata(),'r')
                ax.plot(vcont_obj.contlines[1].get_ydata(),'r')
                ax.axis(ax_lim)
                fig.savefig(os.path.join(save_figs_to, k+'.png'))

        #diams = [objs[k].vconts.get_diameter() for k in keys]
        out = {k:objs[k].vconts.diameter for k in keys}
        if fname is not None:
            if format == 'csv':
               if _with_pandas:
                   out = pd.DataFrame(out)
                   writer = pd.DataFrame.to_csv
               else:
                   writer = io.write_dict_csv
            elif format == 'mat':
               writer = lambda data, name: scipy.io.matlab.savemat(name, data)
            else:
               print("Don't know how to save to format %s"%format)
               writer = lambda data, name: None
            if fname is not None:
                writer(out, fname)
        return out

    def _lutconv(self, frame):
        ccmap = self._ccmap
        if ccmap['i'] is not None:
            i = ccmap['i']
            vmin,vmax = self.clims[i]
            out_frame = (frame[...,i]- 1.0*vmin)/(vmax-vmin)
        else:
            out_frame = np.zeros(frame.shape[:2] + (3,), dtype=np.float32)
            for k,c in enumerate('rgb'):
                if ccmap[c] is not None:
                    i = ccmap[c]
                    f = frame[...,i]
                    vmin,vmax = self.clims[i]
                    out_frame[...,k] = (f - 1.0*vmin)/(vmax-vmin)
        return np.clip(out_frame, 0,1)

    def add_frameshow_hook(self, fn, label):
        self.frame_hooks[label] = fn

    def _get_show_f(self, n):
        Nf = len(self.frame_coll)
        fi = int(n)%Nf
        if fi in self.frame_cache:
            show_f = self.frame_cache[fi]
        else:
            show_f = self._lutconv(self.frame_coll[fi])
            if self._use_cache_flag:
                self.frame_cache[fi] = show_f
        return show_f

    def set_frame_index(self,n):
        Nf = len(self.frame_coll)
        fi = int(n)%Nf
        self.frame_index = fi

        dz,zunits = quantity_to_pair(self.frame_coll.meta['axes'][0])

        if zunits == '_':
            tstr = ''
        else:
            tstr='(%3.3f %s)'%(fi*dz,zunits)
        _title = '%03d '%fi + tstr
        show_f = self._get_show_f(n)

        self.plh.set_data(show_f)
        self.ax1.set_title(_title)
        if self.frame_slider:
            if self.frame_slider.val !=n:
                #print 'updating frame slider'
                self.frame_slider.set_val(n)
        for h in list(self.frame_hooks.values()):
            h(n)
        self.fig.canvas.draw()

    def show_home_frame(self):
        f = self.home_frame
        _title = "Home frame"
        show_f = self._lutconv(f)
        self.plh.set_data(show_f)
        self.ax1.set_title(_title)
        self.fig.canvas.draw()
        self.frame_index = 0
        #if hasattr(self, 'caller'): #called from frame_viewer
        #    self.caller.frame_index = self.frame_index

    def frame_skip(self,event, n=1):
        if not self.event_canvas_ok(event):
            return
        fi = self.frame_index
        key = hasattr(event, 'button') and event.button or event.key
        prev_keys = [4,'4','down','left','p']
        next_keys = [5,'5','up','right','n']
        home_keys =  ['h','q','z']
        known_keys = prev_keys+ next_keys+home_keys
        if key in known_keys:
            if key in home_keys:
                self.show_home_frame()
            else:
                if key in prev_keys:
                    fi -= n
                elif key in next_keys:
                    fi += n
                self.set_frame_index(fi)
        if hasattr(self, 'caller'): #called from frame_viewer
            self.caller.frame_index = self.frame_index

    def connect(self):
        "connect all the needed events"
        if self._verbose:
            print("connecting callbacks to picker")
        cf = self.fig.canvas.mpl_connect
        self.cid = {
            'click': cf('button_press_event', self.on_click),
            'release': cf('button_release_event', self.on_release),
            'motion': cf('motion_notify_event', self.on_motion),
            'scroll':cf('scroll_event',self.frame_skip),
            'type':cf('key_press_event',self.frame_skip),
            'modkey_on':cf('key_press_event', self.on_modkey),
            'key_release':cf('key_release_event', self.on_keyrelease)
            }
    def disconnect(self):
        if hasattr(self, 'cid'):
            if self._verbose:
                print("disconnecting old callbacks")
            for cc in self.cid.values():
                self.fig.canvas.mpl_disconnect(cc)

    def isAreaROI(self, tag):
        return self.roi_objs[tag].roi_type == 'area'
    def isPathROI(self, tag):
        return self.roi_objs[tag].roi_type == 'path'
    def isLineROI(self,tag):
        return isinstance(self.roi_objs[tag], LineScan)

    def get_area_roi_tags(self):
        return sorted(list(filter(self.isAreaROI, self.roi_objs.keys())))

    def get_timeseries(self, rois=None, **zview_kwargs):
        rois = ifnot(rois,
                     sorted(list(filter(self.isAreaROI, self.roi_objs.keys()))))
        return [self.roi_objs[tag].get_zview(**zview_kwargs)    for tag in  rois]
    def pandize(self, data,index=None):
        """If has Pandas loaded, return pandas.DataFrame from data"""
        if _with_pandas:
            data = pd.DataFrame(data,index=index)
        return data
    def get_roi_data(self, only_1d=False,with_diameters=True, use_pandas=True):
        robjs = self.roi_objs
        area_rois = sorted(list(filter(self.isAreaROI, robjs)))
        linescan_rois = sorted(list(filter(self.isLineROI, robjs)))

        rdata = {tag:robjs[tag].get_zview() for tag in area_rois}
        dx = self.frame_coll.meta['axes'][1]
        if with_diameters:
            for r in linescan_rois:
                if robjs[r].has_traced_vessels:
                    rdata[r+'-diam'] = robjs[r].vconts.diameter*dx.value
        if use_pandas:
            rdata = self.pandize(rdata,index=self.active_stack.frame_idx())
        if not only_1d:
            print(linescan_rois)
            acc2d = {r:robjs[r].get_zview() for r in linescan_rois}
            if not use_pandas:
                rdata.extend(acc2d)
            else:
                rdata = [rdata, acc2d]
        return rdata


            
        

    ## def timevec(self):
    ##     dt,Nf = self.dt, self.length()
    ##     return np.arange(0,Nf*dt, dt)[:Nf]

    # def export_roi_signals(self, fname, format='csv', normp=False):
    #     known_formats = ['csv', 'tab', 'pickle', 'hdf', 'mat']
    #     area_rois = sorted(filter(self.isAreaROI, self.roi_objs.keys()))
    #     all_rois = self.roi_tags()
    #     if len(area_rois) == len(all_rois):
    #         ts = self.get_timeseries(normp=normp)
    #         t = self.fseq.frame_idx()
    #         with open(fname, 'w') as fd:
    #             if hasattr(self.fseq, 'ch'):
    #                 out_string = "Channel %d\n" % self.fseq.ch
    #             else:
    #                 out_string = ""
    #             out_string += "Time\t" + '\t'.join(point_rois) + '\n'
    #             for k in xrange(self.length()):
    #                 out_string += "%e\t"%t[k]
    #                 out_string += '\t'.join(["%e"%a[k] for a in ts])
    #                 out_string += '\n'
    #             fd.write(out_string)
    #     else:
    #         acc = [(tag,self.roi_objs[tag].get_zview())
    #                for tag in sorted(self.roi_objs.keys())]
    #         if not '.pickle' in fname:
    #             fname +='.pickle'
    #         pickle.dump(acc, open(fname, 'w'))
    #         if self._verbose:
    #             print "Saved time-views for all rois to ", fname

    def export_roi_signals(self, fname, format='csv'):
        #TODO: multiple color save to csv and tab
        known_formats = ['csv', 'tab', 'pickle', 'hdf', 'mat', 'npy']
        if format not in known_formats:
            print('Pickle.export_roi_signals: unknown save format')
            print('can only save to {}'.format(known_formats))
            return
        def _dict_copy_no_nones(d):
            return {k:(v if v is not None else 'None') for k,v in list(d.items())}
        all_rois = self.roi_tags()
        tv = self.active_stack.frame_idx()
        if format in ['csv', 'tab']:
            # only save 1D signals to csv and tab
            all_rois = list(filter(self.isAreaROI, all_rois))

        all_zv = {t:self.roi_objs[t].get_zview() for t in all_rois}
        if format in ['mat', 'pickle', 'hdf']:
            all_zv['meta'] = _dict_copy_no_nones(self.frame_coll.meta)
            all_zv['roi_props'] = list(map(_dict_copy_no_nones, self.export_rois()))
            #print all_zv['roi_props']
        if format == 'mat':
            # drop constructor functions for mat files
            for rd in all_zv['roi_props']:
                rd.pop('func')
            scipy.io.matlab.savemat(fname, all_zv)

        elif format == 'csv':
            #writer = partial(io.write_dict_csv, index=('time, s',tv))
            io.write_dict_csv(all_zv, fname, index=('time, s', tv))
        elif format == 'tab':
            io.write_dict_tab(all_zv,fname,index=('time,s',tv))
        elif format == 'pickle':
            pickle.dump(all_zv, open(fname, 'wb'))
        elif format == 'npy':
            np.save(fname, all_zv)
        elif format == 'hdf':
            io.write_dict_hdf(all_zv, fname)
        if self._verbose:
            print('Picker: saved time-views for rois to ', fname)
        return

    def show_zview(self, rois = None, **kwargs):
        print('in Picker.show_zview()')

        tx = self.active_stack.frame_idx()
        t0 = tx[self.frame_index]

        if rois is None: rois = self.roi_tags()
        _key = lambda t: self.roi_objs[t].roi_type
        figs = []
        #roi_types = np.unique(map(_key, rois))
        #roi_groups = [(t,[r for r in rois if _key(r)==t]) for t in roi_types]
        roi_groups = itt.groupby(sorted(rois, key=_key), _key)
        for roi_type, roi_tgroup in roi_groups:
            print(roi_type, roi_tgroup)
            roi_tgroup = list(roi_tgroup)

            prefs = self.roi_prefixes(roi_tgroup)
            if roi_type is 'area':
                print(prefs)
                _sh = self.roi_objs[roi_tgroup[0]].get_zview().shape

                fig, axs = plt.subplots(len(prefs),len(_sh) > 1 and _sh[1] or 1, squeeze=False)

                for ax in np.ravel(axs):
                    lh = ax.axvline(t0, color='k', ls='-',lw=0.5)
                    def _cursor(n):
                        try:
                            lh.set_xdata(tx[n])
                            fig.canvas.draw()
                        except:
                            pass
                    self.add_frameshow_hook(_cursor, roi_tgroup[0]+rand_tag())

                figs.append(fig)

                if len(_sh)>1:
                    colors = 'red', 'green', 'blue' #TODO: harmonize with color names in fseq?
                    for ax, c in zip(axs[0,:], colors):
                        ax.set_title(c+' channel', color=c)

                for row, prefix in enumerate(prefs):
                    labels = []
                    matching_tags = [t for t in roi_tgroup if t.startswith(prefix)]
                    for t in matching_tags:
                        roi = self.roi_objs[t]
                        color = roi.color
                        signal = roi.get_zview(**kwargs)
                        if np.ndim(signal) < 2:
                            signal = signal[:,None]
                        for k, ve in enumerate(signal.T):
                            if np.any(ve):
                                ax = axs[row,k]
                                ax.plot(tx, ve, color=color, alpha=0.5, label=t)
                                ## ax.text(0.9,0.9, t,
                                ##         fontsize=14,
                                ##         transform=ax.transAxes,color=color,
                                ##         alpha=1.0,
                                ##         visible=False)
                    for k in range(signal.shape[1]):
                        l = axs[row,k].legend(frameon=False)
                        if l is None: continue
                        for lx in l.legendHandles:
                            lx.set_visible(True)
                            lx.set_linewidth(2)
                            lx.set_alpha(0.5)
                    axs[row,0].set_ylabel(prefix)
                    #DONE: add event that hovering mouse over trace makes its alpha=1

                def _line_highlighter(event):
                    _ax = event.inaxes
                    if _ax is None:
                        return
                    #for line,label in zip(_ax.lines,_ax.texts):
                    for line, label in zip(_ax.lines[1:], _ax.legend_.legendHandles):
                        if line.contains(event)[0]:
                            line.set_alpha(1.0)
                            line.set_linewidth(2)
                            label.set_visible(True)
                            label.set_alpha(1.0)
                            #leg.set_linewidth(2)
                        else:
                            line.set_alpha(0.5)
                            line.set_linewidth(1)
                            label.set_visible(True)
                            label.set_alpha(0.5)                            
                    _ax.figure.canvas.draw()

                # #TODO: may be use on_click event?
                fig.canvas.mpl_connect('motion_notify_event', _line_highlighter)
            elif roi_type is 'path':
                print('path')
                roi_tgroup = list(roi_tgroup)
                slices = [self.roi_objs[t].get_zview(**kwargs)[0] for t in roi_tgroup]
                ncols = int(np.ceil(len(slices)/5.))
                group_maps(list(map(self._lutconv, slices)),
                               ncols=ncols, titles = roi_tgroup,
                               colorbar=(np.ndim(self.active_stack[0])<2))
                figs.append(plt.gcf())
            else:
                print('ui.Picker.show_zview: unknown roi type, %s'%roi_type)
            for f in figs:
                 f.show()
        return figs


    def show_ffts(self, rois = None, **keywords):
        L = self.length()
        dt = self.frame_coll.meta['axes'][0]
        freqs = np.fft.fftfreq(int(L),dt)[1:L/2]
        for x,tag,roi,ax in self.roi_show_iterator_subplots(rois, **keywords):
            y = abs(np.fft.fft(x))[1:L/2]
            ax.plot(freqs, y**2)
        ax.set_xlabel("Frequency, Hz")

    def show_xcorrmap(self, roitag, figsize=(6,6),
                      **kwargs):
        roi =  self.roi_objs[roitag]
        signal = self.get_timeseries([roitag],normp=False)[0]
        xcmap = fnmap.xcorrmap(self.active_stack, signal, corrfn=self._corrfn,**kwargs)
        mask = roi.in_circle(xcmap.shape)
        xshow = np.ma.masked_where(mask,xcmap)
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        vmin, vmax = xshow.min(), xshow.max()
        im = ax.imshow(ndimage.median_filter(xcmap,3), aspect = 'equal', vmin=vmin,
                       vmax=vmax,cmap=_wavelet_cmap)
        plt.colorbar(im, ax=ax)
        ax.set_title('Correlation to %s'%roitag)
        return xcmap

    def show_spectrograms(self, rois = None, freqs = None,
                          wavelet = pycwt.Morlet(),
                          vmin = None,
                          vmax = None,
                          normp= True,
                          **keywords):
        keywords.update({'rois':rois, 'normp':normp})
        dt = self.active_stack.meta['axes'][0].value
        f_s = 1.0/dt
        freqs = ifnot(freqs, self.default_freqs())
        axlist = []
        for x,tag,roi,ax in self.roi_show_iterator_subplots(**keywords):
            utils.wavelet_specgram(x, f_s, freqs,  ax,
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
        from swan import utils as swu
        "Create a figure of a signal, spectrogram and a colorbar"
        if not self.isAreaROI(roitag):
            print("This is not an area ROI, exiting")
            return
        signal = self.get_timeseries([roitag],normp=DFoSD)[0]
        Ns = len(signal)
        dz = self.active_stack.meta['axes'][0]
        zunits = str(dz.unit)        
        dz = dz.value
        f_s = 1/dz
        freqs = ifnot(freqs,self.default_freqs())
        title_string = ifnot(title_string, roitag)
        tvec = self.active_stack.frame_idx()
        L = min(Ns,len(tvec))
        tvec,signal = tvec[:L],signal[:L]
        lc = self.roi_objs[roitag].color
        fig,axlist = swu.setup_axes_for_spectrogram((8,4))
        axlist[1].plot(tvec, signal,'-',color=lc)
        utils.wavelet_specgram(signal, f_s, freqs,  axlist[0], vmax=vmax,
                               wavelet=wavelet,
                               cmap=_wavelet_cmap,
                               cax = axlist[2])
        axlist[0].set_title(title_string)
        #zunits = hasattr(dz, 'unit') and dz.unit or ''
        
        if zunits != '_':
            axlist[1].set_xlabel('time, %s'%zunits)
            axlist[0].set_ylabel('Frequency, Hz')
        return fig

    def show_wmps(self, rois = None, freqs = None,
                  wavelet = pycwt.Morlet(),
                  vmin = None,
                  vmax = None,
                  **keywords):
        "show mean wavelet power spectra"
        keywords.update({'rois':rois, 'normp':True})
        dz = self.active_stack.meta['axes'][0]
        fs = 1.0/dz.value
        freqs = ifnot(freqs, self.default_freqs())
        for x,tag,roi,ax in self.roi_show_iterator_subplots(**keywords):
            cwt = pycwt.cwt_f(x, freqs, fs, wavelet, 'zpd')
            eds = pycwt.eds(cwt, wavelet.f0)
            ax.plot(freqs, np.mean(eds, 1))
        ax.set_xlabel("Frequency, Hz")

    def default_freqs(self, nfreqs = 1024):
        dz = self.active_stack.meta['axes'][0].value
        return np.linspace(4.0/(self.length()*dz),
                           0.5/dz, num=nfreqs)

    def roi_show_iterator_subplots(self, rois = None,
                              **kwargs):
        #rois = ifnot(rois,
        #             sorted(filter(self.isAreaROI, self.roi_objs.keys())))
        #print 'in Picker.roi_show_iterator'
        rois = ifnot(rois, sorted(self.roi_objs.keys()))

        L = len(rois)
        if L < 1: return
        fig,axlist = plt.subplots(len(rois),1,sharey=True,sharex=True,
                                 squeeze = False,
                                 figsize=(5*L,4.5))

        for i, roi_label in enumerate(rois):
            roi = self.roi_objs[roi_label]
            ax = axlist[i,0]
            if self.isAreaROI(roi_label):
                x = roi.get_zview(**kwargs)
            else:
                x,points = roi.get_zview(**kwargs)
            if i == L-1:
                dz = self.active_stack.meta['axes'][0]
                if dz.unit != '_':
                    ax.set_xlabel("time, %s"%dz.unit)
                else:
                    ax.set_xlabel('frames')
            if L == 1:
                ax.set_title(roi_label, color=roi.color,
                             backgroundcolor='w', size='large')
            else:
                ax.set_ylabel(roi_label, color=roi.color,size='large')
            yield x, roi_label, roi, ax

    def roi_prefixes(self, tags=None):
        import re
        if tags is None: tags = self.roi_tags()
        def nonempty(x): return len(x)
        def get_prefix(tag): return list(filter(nonempty, re.findall('[a-z]*',tag)))[0]
        return np.unique(list(map(get_prefix, tags)))


    def show_xwt_roi(self, tag1, tag2, freqs=None,
                     func = pycwt.wtc_f,
                     wavelet = pycwt.Morlet()):
        "show cross wavelet spectrum or wavelet coherence for two ROIs"
        freqs = ifnot(freqs, self.default_freqs())
        dz = self.active_stack.meta['axes'][0]
        self.extent=[0,self.length()*dz.value, freqs[0], freqs[-1]]

        if not (self.isAreaROI(tag1) and self.isAreaROI(tag2)):
            print("Both tags should be for area-type ROIs")
            return

        s1 = self.roi_objs[tag1].get_zview(True)
        s2 = self.roi_objs[tag2].get_zview(True)

        res = func(s1,s2, freqs,1.0/dz,wavelet)

        t = self.active_stack.frame_idx()

        plt.figure();
        ax1= plt.subplot(211);
        roi1,roi2 = self.roi_objs[tag1], self.roi_objs[tag2]
        plt.plot(t,s1,color=roi1.color, label=tag1)
        plt.plot(t,s2,color=roi2.color, label = tag2)
        #legend()
        ax2 = plt.subplot(212, sharex = ax1);
        ext = (t[0], t[-1], freqs[0], freqs[-1])
        ax2.imshow(res, extent = ext, cmap = _wavelet_cmap)
        #self.cone_infl(freqs,wavelet)
        #self.confidence_contour(res,2.0)

    def show_roi_xcorrs(self, corrfn = None, normp = DFoSD,
                        getter = lambda x: x[0]):
        if corrfn == None:
            corrfn = fnmap.stats.pearsonr
        rois = sorted(filter(self.isAreaROI, list(self.roi_objs.keys())))
        ts = self.get_timeseries(normp=normp)
        t = self.active_stack.frame_idx()
        nrois = len(rois)
        if nrois < 2:
            print("not enough rois, pick more")
            return
        out = np.zeros((nrois, nrois))
        #for k1, k2 in lib.allpairs(range(nrois)):
        for k1,k2 in itt.combinations(list(range(nrois)),2):
            coef = getter(corrfn(ts[k1], ts[k2]))
            out[k2,k1] = coef
        f = plt.figure();
        ax = plt.imshow(out, aspect='equal', interpolation='nearest')
        ticks = np.arange(nrois)
        plt.setp(ax.axes, xticks=ticks, yticks=ticks),
        plt.setp(ax.axes, xticklabels=rois, yticklabels=rois)
        plt.colorbar()
        return out

    def show_xwt(self, **kwargs):
        #for p in lib.allpairs(filter(self.isAreaROI, self.roi_objs.keys())):
        for p in itt.combinations(list(filter(self.isAreaROI, list(self.roi_objs.keys()))),2):
            self.show_xwt_roi(*p,**kwargs)
