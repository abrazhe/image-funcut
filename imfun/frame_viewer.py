#!/usr/bin/which python 

import gc # trying to make it work on windows

import wx
import matplotlib as mpl
mpl.use('WXAgg')

from imfun import fseq, fnmap, lib, leica
from imfun import ui as ifui
import glob as Glob


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

def medfilt_fn(npoints):
    mf = signal.medfilt
    return lambda v: mf(v,npoints)

def norm_mf(fs):
    mf = fs.mean_frame()
    return lambda frame: frame/mf - 1.0

class FrameSequenceOpts(HasTraits):
    mfilt7 = lambda v: signal.medfilt(v,7)
    fw_presets = {
        '1. Do nothing' : [],
        '2. Gauss blur' : [fseq.gauss_blur],
        '3. Median filter' : [lambda v:signal.medfilt(v,3)]
        }
    pw_presets = {
        '1. Do nothing' : lambda x:x,
        '2. Norm to SD' : lambda x: x/np.std(x),
        '3. DFoF' : lib.DFoF,
        '4. DoSD' : lib.DoSD,
        '5. Med. filter ' : mfilt7,
        '6. Med. filter + DFoF': lib.flcompose(mfilt7,lib.DFoF),
        '7. Med. filter + DoSD':lib.flcompose(mfilt7,lib.DoSD),
        }
    dt = Float(0.2, label='sampling interval')
    fig_path = Directory("")
    ch = Enum('green', 'red', 'blue', label='Color channel')
    glob = Str('*.tif', label='Glob', description='Image name contains...')
    leica_xml = File('', label='Leica XML', description='Leica properties file')

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
    
    view = View(Group(Item('fig_path', width=400, springy=True, resizable = True,),
                      Item('glob'),
                      Item('leica_xml', width = 100),
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
                      label='Display'),
                width = 800, )
    def __init__(self, parent):
        self.parent = parent

    def _fig_path_changed(self):
        png_pattern = str(self.fig_path + os.sep + '*.png')
        if len(fseq.sorted_file_names(png_pattern)) > 30:
            self.glob = '*.png'
        self.leica_xml = leica.get_xmljob(self.fig_path)
    def _glob_changed(self):
        if len(self.glob) > 5 and '*' in self.glob:
            gl = self.glob.split('*')[0]
            self.leica_xml = leica.get_xmljob(self.fig_path,
                                              gl + "*[0-9].xml")
            
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
        self.fs2_needs_reload = True

    def _fw_trans1_changed(self):
        self.fs2_needs_reload = True
        self.fs.fns = self.fw_presets[self.fw_trans1]

    def _fw_trasn2_changed(self):
        self.fs2.fns = self.fw_presets[self.fw_trans2]

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
            "Can't change colormap because", e

    def set_display_range(self, low, high, fn=lambda x:x):
        self.vmin, self.vmax = map(fn, (low, high))

    def set_percentile_range(self, low, high):
        from scipy import stats
        fi = self.parent.frame_index
        #values = np.asarray(self.parent.frames[1:]).flatten()
        #fn = lambda x: stats.scoreatpercentile(values, x)
        fn = lambda s : self.fs2.data_percentile(s)
        self.set_display_range(low, high, fn)

    def _percentile_btn_fired(self):
        self.set_percentile_range(5,95)
    def _percentile_btn2_fired(self):
        self.set_percentile_range(50,98)

    def get_fs2(self):
        "returns frame sequence after pixelwise transform"
        if self.fs2_needs_reload:
            pw_fn = self.pw_presets[self.pw_trans]
            print pw_fn
            if hasattr(self, 'fs2'):
                del self.fs2
                print "deleted old fs2"
            self.fs2 = self.fs.pw_transform(pw_fn, dtype = np.float32)
            print "fs2 created"
            self.fs2.fns = self.fw_presets[self.fw_trans2]
            print "fs2.fns updated"
            self.fs2_needs_reload = False
        return self.fs2
            
    def reset_fs(self):
        if hasattr(self, 'fs'): del self.fs
        if hasattr(self, 'fs2'): del self.fs2
        gc.collect()
        print "collected garbage"

        pattern = str(self.fig_path + os.sep + self.glob)
        print pattern
        self.fs = fseq.FSeq_imgleic(pattern, ch=color_channels[self.ch],
                                    xmlname = self.leica_xml)
        print "new fs created"
        self.fs.fns = self.fw_presets[self.fw_trans1]
        print "fns1 set"
        self.fs2_needs_reload = True
        self.get_fs2()
        return

    def _load_btn_fired(self):
        if self.parent.frames is not None:
            del self.parent.frames
            self.parent.frames = None
        self.reset_fs()
        self.vmin, self.vmax = self.fs2.data_range()
        self.dt = self.fs.dt
        self.parent._recalc_btn_fired()

class Test(HasTraits):
    fso = Instance(FrameSequenceOpts)
    figure = Instance(Figure, ())
    max_frames = Int(100)
    frame_index = Int(0)
    frames = None
    coords_stat = String()
    time_stat = String()
    recalc_btn = Button("Recalc frames")

    frame_fwd_btn = Button('frame +')
    frame_back_btn = Button('- frame')


    view = View(HSplit(Group(Item('recalc_btn', show_label=False),
                             Item('figure', editor=MPLFigureEditor(),
                                  show_label=False),
                             HGroup(Item('frame_back_btn'),
                                    Item('frame_index',
                                         editor=RangeEditor(low=0,
                                                            high_name='max_frames',
                                                            mode='slider'),
                                         springy = True),
                                    Item('frame_fwd_btn'),
                                    show_labels = False,
                                    padding = 0,)),
                       Item('fso', style='custom'),
                       springy = True,
                       show_labels=False),
                width=1200,
                height=600,
                resizable=True,
                statusbar = [StatusItem('coords_stat'),
                             StatusItem('time_stat')])

    def _frame_fwd_btn_fired(self):
        self.frame_index += 1

    def _frame_back_btn_fired(self):
        self.frame_index -= 1


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
        if len(self.frames) > 0:
            self.pl.set_data(self.frames[self.frame_index])
            self.pl.axes.figure.canvas.draw()
            t = self.frame_index * self.fso.fs.dt
            self.time_stat = "time: %3.3f"%t

    def redraw(self):
        try:
            self.pl.axes.figure.canvas.draw()
        except:
            print "Can't redraw"

    def _recalc_btn_fired(self):
        fs2 = self.fso.get_fs2()
        self.fso.vmin, self.fso.vmax = fs2.data_range()

        if hasattr(fs2, 'data'):
            self.frames = fs2.data
        else:
            self.frames = fs2.as3darray(dtype = np.float32)

        Nf = len(self.frames)
        if hasattr(self, 'picker'):
            self.picker.disconnect()
        self.picker = ifui.Picker(fs2)
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
