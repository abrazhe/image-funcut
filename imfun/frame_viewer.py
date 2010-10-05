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
from matplotlib.backends.backend_wx import NavigationToolbar2Wx

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
        toolbar = NavigationToolbar2Wx(mpl_control)
        sizer.Add(toolbar, 0, wx.EXPAND)
        self.value.canvas.SetMinSize((10,10))
        return panel

class MPLFigureEditor(BasicEditorFactory):
    klass = _MPLFigureEditor

color_channels = {
    'red':0,
    'green':1,
    'blue':2,
    }

from numpy import sin, cos, linspace, pi
import os

class FrameSequenceOpts(HasTraits):
    mfilt7 = lambda v: signal.medfilt(v,7)
    fw_presets = {
        'Do nothing' : [],
        'Gauss blur' : [fseq.gauss_blur],
        'Median filter (3pix)' : [lambda v:signal.medfilt(v,3)]
        }
    pw_presets = {
        'Do nothing' : lambda x:x,
        'Norm to SD' : lambda x: x/np.std(x),
        'Med. filter (7 points)' : mfilt7,
        'Med. filter (7p) + DFoF': lib.flcompose(mfilt7,lib.DFoF),
        'Med. filter (7p) + DoSD':lib.flcompose(mfilt7,lib.DoSD),
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

    load_btn = Button("Load images")

    
    view = View(Group(Item('fig_path'),
                      Item('glob'),
                      Item('ch'),
                      Item('dt'),
                      Item('load_btn', show_label=False),
                      label='Loading'),
                Group(Item('fw_trans1'),
                      Item('pw_trans'),
                      Item('fw_trans2'),
                      label='Post-process'))

class Test(HasTraits):
    fso = Instance(FrameSequenceOpts)
    figure = Instance(Figure, ())
    max_frames = Int(100)
    frame_index = Int(0)
    frames = None
    load_btn = Button("Load images")
    view = View(HSplit(Group(Item('load_btn', show_label=False),
                             Item('figure', editor=MPLFigureEditor(),
                                  show_label=False),
                             Item('frame_index',
                                  editor=RangeEditor(low=0,
                                                     high_name='max_frames',
                                                     mode='slider'))),
                       Item('fso', style='custom'),
                       show_labels=False),
                width=800,
                height=600,
                resizable=True)

    def _figure_default(self):
        figure = Figure()
        self.axes = figure.add_axes([0.05, 0.04, 0.9, 0.92])
        return figure

    def _fso_default(self):
        fso = FrameSequenceOpts()
        return fso

    def _frame_index_changed(self):
        if self.frames:
            self.pl.set_data(self.frames[self.frame_index])
            self.pl.axes.figure.canvas.draw()

    def _load_btn_fired(self):
        fso = self.fso
        pattern = str(fso.fig_path + os.sep + fso.glob)
        print pattern
        self.axes.cla()
        fs = fseq.FSeq_imgleic(pattern,
                               ch=color_channels[self.fso.ch])
        fs.fns = fso.fw_presets[fso.fw_trans1]
        fs = fs.pw_transform(fso.pw_presets[fso.pw_trans])
        fs.fns = fso.fw_presets[fso.fw_trans2]
        self.frames = [fs.mean_frame()] + fs.aslist()
        Nf = len(self.frames)
        self.picker = ifui.Picker(fs)
        _,self.pl = self.picker.start(ax=self.axes)
        self.pl.axes.figure.canvas.draw()
        self.max_frames = Nf-1

def main():
    Test(fso=FrameSequenceOpts()).configure_traits()

if __name__ == "__main__":
    main()
