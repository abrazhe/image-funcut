#!/usr/bin/which python 
import wx
import matplotlib as mpl
mpl.use('WXAgg')

from imfun import fseq, fnmap, lib
from imfun import ui as ifui

import numpy as np

from scipy import signal

from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg  as FigureCanvas
from matplotlib.figure import Figure
#from matplotlib.backends.backend_wx import NavigationToolbar2Wx
from matplotlib.backends.backend_wx import NavigationToolbar2Wx as NToolbar

from enthought.traits.api import *
from enthought.traits.ui.api import *


from enthought.traits.ui.wx.editor import Editor
from enthought.traits.ui.basic_editor_factory import BasicEditorFactory



class _MPLFigureEditor(Editor):
    scrollable  = True
    def init(self, parent):
        self.control = self._create_canvas(parent)
        self.set_tooltip()
    def update_editor(self):
        pass
    def _create_canvas(self, parent):
        """ Create the MPL canvas. """
        # The panel lets us add additional controls.
        panel = wx.Panel(parent, -1, style=wx.CLIP_CHILDREN)
        sizer = wx.BoxSizer(wx.VERTICAL)
        panel.SetSizer(sizer)
        # matplotlib commands to create a canvas
        mpl_control = FigureCanvas(panel, -1, self.value)
        sizer.Add(mpl_control, 1, wx.LEFT | wx.TOP | wx.GROW)
        toolbar = NToolbar(mpl_control)
        sizer.Add(toolbar, 0, wx.EXPAND)
        self.value.canvas.SetMinSize((10,10))
        return panel

class MPLFigureEditor(BasicEditorFactory):
    klass = _MPLFigureEditor

color_channels = {'red':0, 'green':1,'blue':2, }

from numpy import sin, cos, linspace, pi
import os

class FrameSequenceOpts(HasTraits):
    mfilt7 = lambda v: signal.medfilt(v,7)
    fw_presets = {
        '1. Do nothing' : [],
        '2. Gauss blur' : [fseq.gauss_blur],
        '3. Median filter (3pix)' : [lambda v:signal.medfilt(v,3)]
        }
    pw_presets = {
        '1. Do nothing' : lambda x:x,
        '2. Norm to SD' : lambda x: x/np.std(x),
        '3. DFoF' : lib.DFoF,
        '4. DoSD' : lib.DoSD,
        '5. Med. filter (7 points)' : mfilt7,
        '6. Med. filter (7p) + DFoF': lib.flcompose(mfilt7,lib.DFoF),
        '7. Med. filter (7p) + DoSD':lib.flcompose(mfilt7,lib.DoSD),
        }
    dt = Float(0.2, label='sampling interval')
    fig_path = Directory("")
    ch = Enum('green', 'red', 'blue', label='Color channel')
    glob = Str('*.tif', label='Glob', description='Image name contains...')

    fw_trans1 = Enum(*sorted(fw_presets.keys()),
                     label='Framewise transform before')
    pw_trans = Enum(*sorted(pw_presets.keys()), label='Pixelwise transform')

    fw_trans2 = Enum(*sorted(fw_presets.keys()),
                     label='Framewise transform after')
    
    interpolation = Enum(['nearest', 'bilinear', 'bicubic', 'hanning',
                           'hamming', 'hermite', 'kaiser', 'quadric',
                           'gaussian', 'bessel', 'sinc', 'lanczos',
                           'spline16',],
                          label = "Image interpolation")

    colormap = Enum(['gray', 'jet', 'hsv', 'hot'])

    vmax = Float(255)
    vmin = Float(255)

    percentile_btn = Button("5%-95% range")
    percentile_btn2 = Button("50%-98% range")
    load_btn = Button("Load images",)
    
    view = View(Group(Item('fig_path'),
                      Item('glob'),
                      Item('ch'),
                      Item('dt'),
                      Item('load_btn', show_label=False),
                      label='Loading'),
                Group(Item('fw_trans1'),
                      Item('pw_trans'),
                      Item('fw_trans2'),
                      label='Post-process'),
                Group(Item('vmax'),
                      Item('vmin'),
                      Item('percentile_btn',show_label=False),
                      Item('percentile_btn2',show_label=False),
                      Item('interpolation'),
                      Item('colormap'),
                      label='Display'))
    def __init__(self, parent):
        self.parent = parent
    def _vmax_changed(self):
        try:
            self.parent.pl.set_clim((self.vmin, self.vmax))
            self.parent.pl.axes.figure.canvas.draw()
        except Exception as e:
            print "Can't set vmax because", e
    def _vmin_changed(self):
        self._vmax_changed()

    def _dt_changed(self):
        try:
            self.fs.dt = dt
        except Exception as e:
            "Can't reset dt because", e

    def _pw_trans_changed(self):
        self.fs = self.fs.pw_transform(self.pw_presets[self.pw_trans])
        self.fs.fns = self.fw_presets[self.fw_trans2]
        self.vmin, self.vmax = self.fs.data_range()

    def _fw_trans1_changed(self):
        self.fs.fns = self.fw_presets[self.fw_trans1]
        self.vmin, self.vmax = self.fs.data_range()

    def _fw_trans2_changed(self):
        self.fs.fns = self.fw_presets[self.fw_trans2]
        self.vmin, self.vmax = self.fs.data_range()

    def _interpolation_changed(self):
        try:
            self.parent.pl.set_interpolation(self.interpolation)
            self.parent.redraw()
        except Exception as e :
            "Can't change interpolation because", e

    def _colormap_changed(self):
        try:
            self.parent.pl.set_cmap(self.colormap)
            self.parent.redraw()
        except Exception as e :
            "Can't change interpolation because", e

    def set_display_range(self, low, high, fn=lambda x:x):
        self.vmin, self.vmax = map(fn, (low, high))

    def set_percentile_range(self, low, high):
        from scipy import stats
        fi = self.parent.frame_index
        values = np.asarray(self.parent.frames[1:]).flatten()
        fn = lambda x: stats.scoreatpercentile(values, x)
        self.set_display_range(low, high, fn)

    def _percentile_btn_fired(self):
        self.set_percentile_range(5,95)
    def _percentile_btn2_fired(self):
        self.set_percentile_range(50,98)

    def _load_btn_fired(self):
        pattern = str(self.fig_path + os.sep + self.glob)
        print pattern
        fs = fseq.FSeq_imgleic(pattern,
                               ch=color_channels[self.ch])
        fs.fns = self.fw_presets[self.fw_trans1]
        fs = fs.pw_transform(self.pw_presets[self.pw_trans])
        fs.fns = self.fw_presets[self.fw_trans2]
        self.fs = fs
        self.vmin, self.vmax = fs.data_range()
        self.dt = fs.dt
        self.parent._redraw_btn_fired()

class Test(HasTraits):
    fso = Instance(FrameSequenceOpts)
    figure = Instance(Figure, ())
    max_frames = Int(100)
    frame_index = Int(0)
    frames = None
    coords_stat = String()
    time_stat = String()
    redraw_btn = Button("Redraw")
    view = View(HSplit(Group(Item('redraw_btn', show_label=False),
                             Item('figure', editor=MPLFigureEditor(),
                                  show_label=False),
                             Item('frame_index',
                                  editor=RangeEditor(low=0,
                                                     high_name='max_frames',
                                                     mode='slider'))),
                       Item('fso', style='custom'),
                       show_labels=False),
                width=1000,
                height=600,
                resizable=True,
                statusbar = [StatusItem('coords_stat'),
                             StatusItem('time_stat')])

    def _figure_default(self):
        figure = Figure()
        self.axes = figure.add_axes([0.05, 0.04, 0.9, 0.92])
        return figure

    def update_status_bar(self, event):
        if event.inaxes:
            x, y = event.xdata, event.ydata
            self.coords_stat = "x,y: %3.3f,%3.3f,"%(x,y)
            

    def _fso_default(self):
        fso = FrameSequenceOpts(self)
        return fso

    def _frame_index_changed(self):
        if self.frames:
            self.pl.set_data(self.frames[self.frame_index])
            self.pl.axes.figure.canvas.draw()
            t = self.frame_index * self.fso.fs.dt
            self.time_stat = "time: %3.3f"%t

    def redraw(self):
        try:
            self.pl.axes.figure.canvas.draw()
        except:
            print "Can't redraw"

    def _redraw_btn_fired(self):
        fs = self.fso.fs
        self.frames = [fs.mean_frame()] + fs.aslist()
        Nf = len(self.frames)
        self.picker = ifui.Picker(fs)
        self.axes.cla()
        _,self.pl = self.picker.start(ax=self.axes, legend_type='axlegend',
                                      cmap = self.fso.colormap,
                                      interpolation = self.fso.interpolation)
        self.pl.axes.figure.canvas.draw()
        self.axes.figure.canvas.mpl_connect('motion_notify_event',
                                            self.update_status_bar)
        self.frame_index = 0
        self.max_frames = Nf-1


def main():
    Test().configure_traits()

if __name__ == "__main__":
    main()
