#!/usr/bin/which python 

import gc # trying to make it work on windows

import wx
import matplotlib as mpl
mpl.use('WXAgg')

from functools import partial


from scipy import stats, signal, ndimage
from functools import partial # curry-like function
import numpy as np
import pylab


from imfun import fseq, fnmap, lib, leica, bwmorph, atrous
from imfun import filt
from imfun import ui as ifui
import glob as Glob


from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg  as FigureCanvas
from matplotlib.figure import Figure
#from matplotlib.backends.backend_wx import NavigationToolbar2Wx
from matplotlib.backends.backend_wx import NavigationToolbar2Wx as NToolbar

from traits.api import *
from traitsui.api import *


from traitsui.menu \
     import Action, CloseAction, Menu, MenuBar, OKCancelButtons, Separator

from traitsui.wx.editor import Editor
from traitsui.basic_editor_factory import BasicEditorFactory


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

def find_objects(binarr,size_threshold=20):
    labels, nlabels = ndimage.label(binarr)
    objects =  ndimage.find_objects(labels)
    size_count = ndimage.sum(binarr, labels, range(nlabels))
    for k,o in enumerate(objects):
        if size_count[k] < size_threshold:
            x = np.where(labels[o] == k+1)
            labels[x] = 0
    return ndimage.label(labels)[0]
        
    

class GWOpts(HasTraits):
    pw_presets = {'4. identity' : lambda x,normL=110:x,
                  '3. norm to SD' : lambda x, normL=110: x/np.std(x[:normL]),
                  '2. DF/F' : lib.DFoF,
                  '1. DF/SD' : lib.DFoSD,}
    normL = Int(250, label="N baseline frames")
    pw_func = Enum(*sorted(pw_presets.keys()), label='Pixel-wise')
    tmedian_k = Enum([5]+range(1,13,2), label='median filter kernel')
    gauss_size = Range(1.0, 20.0, 1.0, label='Gauss sigma (after)')
    nclose = Range(0,5,1, 'iterations of 3D binary closing')
    nopen = Range(0,5,1,  'iterations of 3D binary opening')
    sigma_thresh = Range(0.0, 10.0, 1.5,
                         label='Binarization threshold, x S.D.')
    size_threshold = Range(1,2000,60,label='Volume threshold')
    do_labels = Bool(True,info_text="Try to segment binary?")
    run_btn = Button('Run')
    roiexport_file = File('', label='Save ROIs to')
    roiexport_btn = Button('Export ROIs')
    view = View(Group(Item('normL'),
                      Item('pw_func'),
                      Item('tmedian_k'),
                      Item('gauss_size'),
                      show_border=True,
                      label='Pre-process'),
                Group(Item('sigma_thresh'),
                      Item('nopen'),
                      Item('nclose'),
                      show_border=True,
                      label = 'Binarize',),
                Group(Item('size_threshold'),
                      Item('do_labels', label='Do labels?'),
                      show_border=True,
                      label = 'Segment',),
                Item('run_btn', show_label=False),
                Group(Item('roiexport_file', label='Save rois to'),
                      Item('roiexport_btn', show_label=False),
                      show_border=True,
                      label='Exporing'),
                )
    def __init__(self, parent):
        self.fso = parent
    def _run_btn_fired(self):
        if hasattr(self.fso, 'fs'):
            seq = self.fso.fs
            fn1 = partial(self.pw_presets[self.pw_func],
                          normL=self.normL)
            if self.tmedian_k < 3:
                pwfn = fn1
            else:
                pwfn = lib.flcompose2(fn1,
                                      medfilt_fn(self.tmedian_k))
            seq1 = seq.pw_transform(pwfn)
            seq1.fns = [partial(filt.gauss_blur, size=self.gauss_size)]
            arr = seq1.as3darray()     # new array with spatially-smoothed data
            sds = float(np.std(arr))   # standard deviation in all data
            binarr = arr > sds*self.sigma_thresh
            if self.nopen > 0:
                binarr = ndimage.binary_opening(binarr, iterations=self.nopen)
            if self.nclose >0:
                binarr = ndimage.binary_closing(binarr, iterations=self.nopen)
            if self.do_labels:
                objects = find_objects(binarr,self.size_threshold)
                out = np.ma.array(objects, mask=objects==0)
                seq2 = fseq.FSeq_arr(out)
                cmap = 'hsv'
            else:
                seq2 = fseq.FSeq_arr(binarr)
                cmap = 'jet'
            self.fso.get_fs = lambda : seq2
            self.fso.colormap = cmap
            self.fso.parent._recalc_btn_fired()
    def _roiexport_btn_fired(self):
        print "!!!!"
        picker = self.fso.parent.picker
        #self.roiexport_file.configure_traits()
        #picker.save_rois(self.roiexport_file)
        f = file(self.roiexport_file,'w')
        for c in self.fso.parent.picker.roi_objs.values():
            if isinstance(c, ifui.CircleROI):
                f.write("%s: C=(%3.3f,%3.3f) R=%3.3f\n"%\
                        (c.tag, c.obj.center[0], c.obj.center[1],
                         c.obj.radius))
        f.close()                    
        pass

class FrameSequenceOpts(HasTraits):
    mfilt7 = lambda v: signal.medfilt(v,7)
    tmedian_k = Enum([5]+range(1,13,2), label='median filter kernel')
    normL = Int(250, label="N baseline frames")
    fw_presets = {
        '1. Do nothing' : [],
        '2. Gauss blur' : [filt.gauss_blur],
        '3. Median filter' : [lambda v:signal.medfilt(v,3)]
        }
    pw_presets = {
        '01. Do nothing' : lambda x:x,
        '02. Norm to SD' : lambda x: x/np.std(x),
        '03. DF/F' : lib.DFoF,
        '04. DF/sigma' : lib.DFoSD,
	'05. DF/F with detrend': atrous.DFoF,
	'06. DF/sigma with detrend': atrous.DFoSD,	
        '07. Med. filter ' : mfilt7,
        '08. Med. filter -> DF/F': lib.flcompose(mfilt7,lib.DFoF),
        '09. DF/F -> Med. filter': lib.flcompose(lib.DFoF, mfilt7),
        '10. Med. filter -> DF/SD':lib.flcompose(mfilt7,lib.DFoSD),
        '11. DF/SD -> Med. filter':lib.flcompose(lib.DFoSD, mfilt7),
        }
    gw_opts = Instance(GWOpts)
    dt = Float(0.2, label='sampling interval')
    fig_path = Directory("")
    ch = Enum('green', 'red', 'blue', label='Color channel')
    glob = Str('*_t*.tif', label='Glob', description='Image name contains...')
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

    colormap = Enum(['gray', 'jet', 'hsv', 'hot','winter','spring'])

    vmax = Float(255)
    vmin = Float(255)
    
    limrange = lambda x: range(5,x-5,5)+range(x-6,x+2,2)

    low_percentile = Enum(limrange(98),label='Low')
    high_percentile = Enum(limrange(98)[::-1],label='High')
    apply_percentile = Button("Apply")


    low_sd_lim = Enum(range(11),label='Low')
    high_sd_lim = Enum(range(11)[::-1],label='High')
    apply_sd_lim = Button("Apply")

    export_movie_filename = File('', description='where to save the movie')
    export_fps = Float(25.0)
    export_start = Int(0)
    export_stop = Int(-1)


    _export_rois_dict = File()
    _load_rois_dict = File()
    export_rois_dict_btn = Button('Export current ROIs')
    load_rois_dict_btn = Button('Load ROIs from file')
    _export_timeseries_file = File()
    export_timeseries_btn = Button('Export timeseries from ROIs')

    
    reset_range_btn = Button("Set")
    load_btn = Button("Load images")
    export_btn = Button('Export movie')

    load_rois_dict_view = View(
     	Item('_load_rois_dict'), 
     	buttons = OKCancelButtons,
     	kind = 'livemodal',
     	width = 600,
     	title = "Load ROIs from file",
     	resizable = True,)
    

    export_rois_dict_view = View(
     	Item('_export_rois_dict'), 
     	buttons = OKCancelButtons,
     	kind = 'livemodal',
     	width = 600,
     	title = "Export current ROIs to a dictionary",
     	resizable = True,)

    export_timeseries_view = View(
     	Item('_export_timeseries_file'), 
     	buttons = OKCancelButtons,
     	kind = 'livemodal',
     	width = 600,
     	title = "Export timeseries from current ROIs",
     	resizable = True,)

    

    def default_traits_view(self):
	## has to be a method so we can declare views for dialogs !?
	view = View(
	    Group(
		Group('fig_path', 'glob',
		      Item('leica_xml', width = 100),
		      'ch', 'dt',
		      Item('load_btn', show_label=False),
		      label = 'Frame sequence',
		      show_border=True),
		Group(Item('load_rois_dict_btn',show_label=False),
		      label = 'ROIs',
		      show_border=True),
		label='Loading'),
	    Group(Item('gw_opts', show_label=False,style='custom'),
		  show_border=False,
		  label='GW'),
	    Group('fw_trans1', 'pw_trans', 'fw_trans2',
		  label='Post-process'),
	    Group(Group(Item('low_percentile'),
			Item('high_percentile'),
			Item('apply_percentile', show_label=False),
			label='Percentile',
			show_border=True,
			orientation='horizontal'),
		  Group(Item('low_sd_lim'),
			Item('high_sd_lim'),
			Item('apply_sd_lim', show_label=False),
			label='%SD',
			show_border=True,
			orientation='horizontal'),
		  HSplit(Item('vmax'),
			 Item('vmin'),
			 Item('reset_range_btn',show_label=False),
			 show_border=True,
			 label='Limits',
			 springy=True),
		  HSplit(Item('interpolation'),
			 Item('colormap'),
			 show_border=True,
			 label='Matplotlib'),
		  label='Display'),
	    Group(Group(Item('export_movie_filename',label='Export to'),
			Item('export_fps', label='Frames/sec'),
			Item('export_start', label='Start frame'),
			Item('export_stop', label='Stop frame'),
			Item('export_btn', show_label=False),
			show_border=True,
			label='Movie'),
		  Group('export_rois_dict_btn',
			'export_timeseries_btn',
			show_border=True,
			show_labels=False,
			label='ROIs'),
		  label="Export"),
	    width = 800, )
	return view
    def __init__(self, parent):
        self.parent = parent
        self.get_fs = self.get_fs2

    def _fig_path_changed(self):
        png_pattern = str(self.fig_path + os.sep + '*.png')
        if len(fseq.sorted_file_names(png_pattern)) > 30:
            self.glob = '*_t*.png'
        self.leica_xml = leica.get_xmljob(self.fig_path)

    def _glob_changed(self):
        if len(self.glob) > 5 and '*' in self.glob:
            gl = self.glob.split('*')[0]
            self.leica_xml = leica.get_xmljob(self.fig_path,
                                              gl + "*[0-9].xml")
            
    def _vmax_changed(self):
        print "setting levels"
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

    def _fw_trans2_changed(self):
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

    def set_display_range(self, low, high, fn=lambda x:float(x)):
        self.vmin, self.vmax = map(fn, (low, high))

    def _apply_sd_lim_fired(self):
        values = np.asarray(self.parent.frames[1:]).flatten()
        sd = np.std(values)
        self.set_display_range(sd*self.low_sd_lim,
                               sd*self.high_sd_lim)


    def _apply_percentile_fired(self):
        fn = lambda s : self.fs2.data_percentile(s)
        self.set_display_range(self.low_percentile, self.high_percentile, fn)

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

    def _reset_range_btn_fired(self):
        self.vmin, self.vmax = self.fs2.data_range()

            
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
	print 'fs2 set'
        return

    def _gw_opts_default(self):
        gw_opts = GWOpts(self)
        return gw_opts


    def _export_btn_fired(self):
        seq = self.get_fs2()
        seq.export_mpeg(self.export_movie_filename,
                        fps = self.export_fps,
			start = self.export_start,
			stop = self.export_stop,
                        cmap=self.colormap,
                        interpolation = self.interpolation,
                        vmin = self.vmin,
                        vmax=self.vmax)

    def _load_rois_dict_btn_fired(self):
	ui = self.edit_traits(view='load_rois_dict_view')
	if ui.result == True:
	    print self._load_rois_dict
	    picker = self.parent.picker
	    picker.load_rois(self._load_rois_dict)


    def _export_rois_dict_btn_fired(self):
	print "in _export_rois_dict_btn_fired"
	ui = self.edit_traits(view='export_rois_dict_view')
	if ui.result == True:
	    print self._export_rois_dict
	    picker = self.parent.picker
	    picker.save_rois(self._export_rois_dict)

    def _export_timeseries_btn_fired(self):
	print "_export_timeseries_btn_fired"
	ui = self.edit_traits(view='export_timeseries_view')
	if ui.result == True:
	    print self._export_timeseries_file
	    picker = self.parent.picker
	    picker.save_time_series_to_file(self._export_timeseries_file)
	
    def _load_btn_fired(self):
        if self.parent.frames is not None:
            del self.parent.frames
            self.parent.frames = None
        self.reset_fs()
	print 'reset_fs done'
        self.vmin, self.vmax = self.fs2.data_range()
	print 'vmin,vmax done'
        self.dt = self.fs.dt
	print 'dt done'
        self.parent._recalc_btn_fired()
	print 'recalc'

class FrameViewer(HasTraits):
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
                title= "Frame viewer",
                statusbar = [StatusItem('coords_stat'),
                             StatusItem('time_stat')])

    def _save_rois(self):
        pass
    

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
        fs2 = self.fso.get_fs()

        if hasattr(fs2, 'data'):
            self.frames = fs2.data
        else:
            self.frames = fs2.as3darray(dtype = np.float32)

        self.frames = np.ma.array(self.frames, mask=np.isnan(self.frames))    

        vl,vh = self.frames.min(), self.frames.max() 
        print "vl,vh:", vl, vh
        self.fso.vmin, self.fso.vmax = float(vl), float(vh)

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
    FrameViewer().configure_traits()

if __name__ == "__main__":
    main()
