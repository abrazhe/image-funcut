import wx
import matplotlib as mpl
mpl.use('WXAgg')

from imfun import fseq


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

from numpy import sin, cos, linspace, pi
import os
class Test(HasTraits):
    figure = Instance(Figure, ())
    max_frames = Int(100)
    frame_index = Int(0)
    frames = None
    fig_dir = Directory("")
    load_btn = Button("Load images")
    color_channel = Enum([0,1,2])
    glob = Str()
    view = View(Group(Group(Item('fig_dir'),
                            Item('glob'),
                            Item('color_channel'),
                            Item('load_btn', show_label=False)),
                      Item('figure', editor=MPLFigureEditor(),
                           show_label=False),
                      Item('frame_index',
                           editor=RangeEditor(low=0,
                                              high_name='max_frames',
                                              mode='slider'))),
                width=800,
                height=600,
                resizable=True)
    
    def __init__(self):
        super(Test, self).__init__()
        self.axes = self.figure.add_subplot(111)

    def _frame_index_changed(self):
        if self.frames:
            self.pl.set_data(self.frames[self.frame_index])
            self.figure.canvas.draw()

    def _load_btn_fired(self):
        pattern = self.fig_dir + os.sep + self.glob
        print pattern
        self.fseq = fseq.FSeq_img(pattern, ch=self.color_channel)
        self.frames = self.fseq.aslist()
        Nf = len(self.frames)
        self.pl = self.axes.imshow(self.fseq.mean_frame(), aspect='equal', cmap='gray')
        self.figure.canvas.draw()
        self.max_frames = Nf

def main():
    Test().configure_traits()

if __name__ == "__main__":
    main()
