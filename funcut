#!python

#import gc # trying to make it work on windows

import wx
import matplotlib as mpl
mpl.use('WXAgg')

from scipy import stats, signal, ndimage
#from functools import partial # curry-like function
import numpy as np

from imfun import fseq, lib, atrous
from imfun import filt
from imfun import ui as ifui
from imfun import mes
#import glob as Glob


#from matplotlib.backends.backend_agg import FigureCanvasAgg #canvas
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg  as FigureCanvas
from matplotlib.figure import Figure
#from matplotlib.backends.backend_wx import NavigationToolbar2Wx
from matplotlib.backends.backend_wx import NavigationToolbar2Wx as NToolbar

from traits.api import *
from traitsui.api import *


from traitsui.menu \
     import Action, CloseAction, Menu, MenuBar, OKCancelButtons, Separator

from traitsui.wx.editor import Editor
#from traitsui.wx.themed_vertical_notebook_editor import ThemedVerticalNotebookEditor
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

color_channels = {'all':None,'red':0, 'green':1,'blue':2}

import os

def medfilt_fn(npoints):
    mf = signal.medfilt
    return lambda v: mf(v,npoints)

def norm_mf(fs):
    mf = fs.mean_frame()
    return lambda frame: frame/mf - 1.0

#class RigidBodyRegister(HasTraits):
#    name = "Rigid body translation motion stab"
#    domain = "temporal"
#    def apply(self, frames):
#        return 

class LoadMotionStab(HasTraits):
    name = 'Load and apply motion stab recipe'
    domain='temporal'
    path = File()
    n_cpu = Int(4)
    view = View('path', 'n_cpu')
    def apply(self, frames):
        if self.path:
            warps = opflowreg.load_recipe(self.path)
            return opflowreg.apply_warps(warps, frames, self.n_cpu)
        else:
            raise NameError("Registration recipe file is not provided")

class CreateMotionStab(HasTraits):
    name = 'Create and apply motion stab recipe'
    domain='temporal'
    save_recipe_to = File()
    n_cpu = Int(4)
    reg_type = Enum('template','recursive')
    reg_pipeline=Enum('translations',
                      'translations->affine',
                      'translations->homography',
                      'Greenberg-Kerr',
                      'translations->softmesh')
    view = View('save_recipe_to', 'reg_type', 'reg_pipeline', 'n_cpu')
    def apply(self, frames):
        registrators = opflowreg.RegistrationInterfaces()
        if self.reg_type == 'template':
            tstart = len(frames)/2
            tstop = min(len(frames),tstart+50)
            template = np.max(frames[tstart:tstop],axis=0)
            def register_stack(stack, registrator):
                return opflowreg.register_stack_to_template(stack,template,registrator, njobs=self.n_cpu)
        elif self.reg_type == 'recursive':
            def register_stack(stack, registrator):
                return opflowreg.register_stack_recursive(stack,registrator)[1]
        else:
            raise NameError("Unknown registration type")
        # TODO: below is just crazy. has to be made neat later
        reg_dispatcher = {'affine':registrators.affine,
                          'homograhy':registrators.homography,
                          'translations':registrators.rigidbody,
                          'Greenberg-Kerr':registrators.greenberg_kerr,
                          'softmesh':registrators.softmesh}
        operations = self.reg_pipeline.split('->')
        newframes = frames
        warp_history = []
        for movement_model in operations:
            warps = register_stack(newframes, reg_dispatcher[movement_model])
            warp_history.append(warps)
            newframes = opflowreg.apply_warps(warps, newframes, njobs=self.n_cpu)
        final_warps = [lib.flcompose(*warpchain) for warpchain in zip(*warp_history)]
        if self.save_recipe_to:
            opflowreg.save_recipe(final_warps, self.save_recipe_to)
            print 'saved motions stab recipe to %s'%self.save_recipe_to
        return newframes

                



from imfun import opflowreg

class KalmanStackFilter(HasTraits):
    name = 'Kalman stack filter'
    domain='temporal'
    seed = Enum("mean", "first")
    gain = Float(0.5)
    var = Float(0.05)
    view = View('seed', 'gain', 'var')
    def apply(self, frames):
        return filt.kalman_stack_filter(frames, seed=self.seed, gain=self.gain,var=self.var)

class SpatialGaussFilter(HasTraits):
    name = 'Spatial Gauss filter'
    domain='spatial'
    sigma = Float(1.0)
    view = View('sigma')
    def apply(self, f):
        return ndimage.gaussian_filter(f, self.sigma)

class SpatialMedianFilter(HasTraits):
    name = 'Spatial Median filter'
    domain='spatial'
    size = Float(3)
    view = View('size')
    def apply(self, f):
        return signal.medfilt(f, self.size)
    
class NormalizeDFoF(HasTraits):
    name='DF/F norm.'
    domain='temporal'
    detrend_level = Int(0)
    use_first_N_points = Int(-1)
    view = View(Item('detrend_level'),
                Item('use_first_N_points',
                     enabled_when='detrend_level == 0'))
    def apply(self, frames):
        if self.detrend_level>0:
            return atrous.DFoF(frames, level=self.detrend_level,
                               axis=0)
        else:
            return lib.DFoF(frames, normL=self.use_first_N_points)
    
    

class FrameSequenceOpts(HasTraits):
    _verbose=Bool(True)

    internal_state_flags = {'fig_path_just_changed':False}

    self.fs = Instance(fseq.FrameSequence) # 'original' frame sequence
    self.fs2 = Instance(fseq.FrameSequence) # 'processed' frame sequence

    linescan_scope = Range(0,500,0, label='Linescan half-range')
    linescan_width = Int(3, label="Linecan linewidth")

    pipeline = List()
    inventory_dict = {p.name:p for p in
                      [LoadMotionStab,
                       CreateMotionStab,
                       KalmanStackFilter,
                       SpatialGaussFilter,
                       SpatialMedianFilter,
                       NormalizeDFoF]}
    
    inventory = Enum(sorted(inventory_dict.keys()))
    add_to_pipeline_btn = Button(label='Add to pipeline')
    clear_pipeline_btn = Button(label='Clear pipeline')

    dt = Float(1, label='frame interval')
    dtunits = Str("",label='units')
    masterpath = Directory(os.getcwd())
    fig_path = File()
    fig_full_path = ""

    record = Str('1')
    avail_records = List(['1'])
    record_enabled = Bool(False)

    ch = Enum('all', 'red', 'green', 'blue', label='Color channel')

    #glob = Str('*_t*.tif', label='Pattern', description='Image name contains...')
    #glob_enabled = Bool(True)
 
    interpolation = Enum(['nearest', 'bilinear', 'bicubic', 'hanning',
                          'hamming', 'hermite', 'kaiser', 'quadric',
                          'gaussian', 'bessel', 'sinc', 'lanczos',
                          'spline16',],
                          label = "Image interpolation")

    colormap = Enum(['gray', 'jet', 'hsv', 'hot','winter','spring','spectral'])

    vmax = Float(255)
    vmin = Float(0)
    
    #limrange = lambda : [0.5, 1, 2] +range(5,96,10)+[98,99,99.5]
    def limrange(): return[0.5, 1, 2] +range(5,96,10)+[98,99,99.5]

    low_percentile = Enum(limrange(),label='Low')
    high_percentile = Enum(limrange()[::-1],label='High')
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
    _export_timeseries_file = File()
    _export_vessel_diameters_file = File()

    also_save_figures = Bool(False)
    diameter_save_format = Enum(["csv", "mat"])
    
    export_vessel_diameters_btn = Button("Export vessel diameters")
    export_rois_dict_btn = Button('Export current ROIs')
    load_rois_dict_btn = Button('Load ROIs from file')
    
    export_timeseries_btn = Button('Export timeseries from ROIs')
    show_all_timeseries_btn = Button('Show all timeseries')
    trace_all_vessels_btn = Button('Vessel Contours for all LineROIs')
    drop_all_rois_btn = Button("Drop all ROIs")
    
    reset_range_btn = Button("Reset")
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

    export_vessel_diameters_view = View(
     	Item('_export_vessel_diameters_file'), 
     	buttons = OKCancelButtons,
     	kind = 'livemodal',
     	width = 600,
     	title = "Export traced vessel diameters from LineScan ROIs",
     	resizable = True,)


    def default_traits_view(self):
	"""Default view for FrameSequenceOpts"""
	## has to be a method so we can declare views for dialogs !?
	view = View(
	    Group(
		Group(Item('masterpath', label='Directory'),
                      Item('fig_path'),
                      #Item('glob', enabled_when='glob_enabled is True'),
                      Item('record',
                           editor=EnumEditor(name='avail_records'),
                           enabled_when='record_enabled is True',
                           style='simple'),
		      'ch',
                      HGroup('dt','dtunits'),
		      Item('load_btn', show_label=False),
		      label = 'Frame sequence',
		      show_border=True),
		Group(HGroup(Item('linescan_width'), Item('linescan_scope')),
		      Item('show_all_timeseries_btn',show_label=False),
                      Item('trace_all_vessels_btn',show_label=False),
		      Item('load_rois_dict_btn',show_label=False),
                      Item('drop_all_rois_btn',show_label=False),
		      label = 'ROIs',
		      show_border=True),
		label='Open'),
	    Group(Group(Item('pipeline', show_label=False, style='custom',
                             editor=ListEditor(use_notebook=True,
                                               page_name = '.name',
                                               deletable=True)),
                        HGroup(Item('inventory',show_label=False),
                               Item('add_to_pipeline_btn',show_label=False),
                               Item('clear_pipeline_btn',show_label=False)),
                        label='Pipeline',show_border=True),
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
                        Group(
                            Item('export_vessel_diameters_btn',show_label=False),
                            'also_save_figures',
                            'diameter_save_format',
                            show_border=True),
			show_border=True,
			show_labels=False,
			label='ROIs'),
		  label="Export"),
	    width = 800, )
	return view
    def __init__(self, parent):
        self.parent = parent
        self.get_fs = self.get_fs2
        #self.masterpath = masterpath

    def _masterpath_changed(self):
        os.chdir(self.masterpath)
        if not self.internal_state_flags['fig_path_just_changed']:
            self.fig_path = ""
            self.record = "1"
            self.record_enabled = False

    def _fig_path_changed(self):
        print "-----", self.fig_path
        ext = self.fig_path.split('.')[-1].lower()
        if not self.internal_state_flags['fig_path_just_changed']:
            if ext == 'mes':
                vars = mes.load_file_info(self.fig_path)
                vars = [v for v in vars if v.is_supported()]
                self.avail_records = map(repr, vars)
                self.record = self.avail_records[0]
                self.record_enabled=True
            self.fig_full_path = self.fig_path
        self.internal_state_flags['fig_path_just_changed'] = True
        self.masterpath = os.path.abspath(os.path.dirname(self.fig_path))
        self.fig_path = os.path.split(self.fig_path)[1]
        self.internal_state_flags['fig_path_just_changed'] = False
        
        
            
    ## def _record_changed(self):
    ##     if self.fig_path.split('.')[-1].lower() == 'mes':
    ##         print self.record
            
    ## def _glob_changed(self):
    ##     if len(self.glob) > 5 and '*' in self.glob:
    ##         gl = self.glob.split('*')[0]
            
    def _vmax_changed(self):
        print "setting levels"
        if hasattr(self.parent, 'pl'):
            try:
                self.parent.pl.set_clim((self.vmin, self.vmax))
                self.parent.pl.axes.figure.canvas.draw()
            except Exception as e:
                print "Can't set vmax because", e
    def _vmin_changed(self):
        self._vmax_changed()

    def _ch_changed(self):
        if hasattr(self, 'fs'):
            self.fs.ch = color_channels[self.ch]
        

    def _dt_changed(self):
        try:
            self.fs.meta['axes']['scale'][0] = self.dt
        except Exception as e:
            "Can't reset frame interval because", e

    def _dtunits_changed(self):
        try:
            self.fs.meta['axes']['scale'][1] = self.dtunits
        except Exception as e:
            "Can't reset frame interval units because", e


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

    def _add_to_pipeline_btn_fired(self):
        filt = self.inventory_dict[self.inventory]
        self.pipeline.append(filt())
    def _clear_pipeline_btn_fired(self):
        self.pipeline = []
        

    def _apply_sd_lim_fired(self):
        values = np.asarray(self.parent.frames[1:]).flatten()
        sd = np.std(values)
        self.set_display_range(sd*self.low_sd_lim,
                               sd*self.high_sd_lim)

    def _apply_percentile_fired(self):
        #fn = lambda s : self.fs2.data_percentile(s)
        def fn(s): return self.fs2.data_percentile(s)
        self.set_display_range(self.low_percentile, self.high_percentile, fn)
    
    def _drop_all_rois_btn_fired(self):
        print "in drop_all_rois"
        if hasattr(self.parent, 'picker'):
            if self._verbose: print "Dropping all rois..."
            self.parent.picker.drop_all_rois()

    def _pipeline_items_changed(self):
        self._fs2_needs_reload=True

    def get_fs2(self):
        "returns frame sequence after pipeline processing"
        if self._fs2_needs_reload:
            out = self.fs
            out.fns = []
            for f in self.pipeline:
                if f.domain == 'spatial':
                    out.fns.append(f.apply)
                elif f.domain == 'temporal':
                    out = fseq.open_seq(f.apply(out.as3darray()))
                else:
                    print 'unknown filter domain'
            if hasattr(self.fs, 'meta'):
                out.meta['axes'] = self.fs.meta['axes'].copy()
            self.fs2 = out
            self._fs2_needs_reload = False
        return self.fs2

    def _show_all_timeseries_btn_fired(self):
	print 'in show_all_timeseries_btn_fired'
	self.parent.picker.show_zview()
	pass
    def _trace_all_vessels_btn_fired(self):
        self.parent.picker.trace_vessel_contours_in_all_linescans(hwidth=self.linescan_width)

    def _reset_range_btn_fired(self):
        if len(self.fs2.shape())<3:
            r = self.fs2.data_percentile((0.001, 99.99))
            # it makes sense to set vmin,vmax values only
            # if there is one color channel
            self.vmin, self.vmax = r

            
    def reset_fs(self):
        #if hasattr(self, 'fs'): del self.fs
        #if hasattr(self, 'fs2'): del self.fs2

        #ext = self.fig_path.split('.')[-1]
        #if ext in ['mes', 'mlf']:
        #    self.glob = ""

        #if len(self.glob) > 0:
        #    path = os.sep.join(self.fig_full_path.split(os.sep)[:-1])
        #    #path = self.fig_path
        #    print path
        #    path = str(path + os.sep + self.glob)
        #else:
        #    path = str(self.fig_full_path)

        path = self.fig_full_path
        if self._verbose:
            print path

        ext = path.split('.')[-1].lower()

        if ext == 'mes': # in case of mes, self.record is repr, not record name
            record = self.record.split(' ')[0][1:]
        else:
            record = self.record

        self.fs = fseq.open_seq(path, record=record, ch=color_channels[self.ch])

        if self._verbose:
            print "new fs created"
        #self.fs.fns = self.fw_presets[self.fw_trans1]
        if self._verbose:
            print "fns1 set"
        self._fs2_needs_reload = True
        self.get_fs2()
        if self._verbose:
            print 'fs2 set'
        return

    def _export_btn_fired(self):
        seq = self.get_fs2()
        seq.export_movie_anim(self.export_movie_filename,
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
	    picker.export_rois(self._export_rois_dict)

    def _export_timeseries_btn_fired(self):
	print "_export_timeseries_btn_fired"
	ui = self.edit_traits(view='export_timeseries_view')
	if ui.result == True:
	    print self._export_timeseries_file
	    picker = self.parent.picker
	    picker.save_time_series_to_file(self._export_timeseries_file)

    def _export_vessel_diameters_btn_fired(self):
        print "_export_vessel_diameters_btn_fired"
        print 
        ui = self.edit_traits(view='export_vessel_diameters_view')
	if ui.result == True:
            name = self._export_vessel_diameters_file
            print 'Picked file name:', name
            p = self.parent.picker
            wd = None
            if self.also_save_figures:
                wd = os.path.abspath(os.path.dirname(name))
                self.parent.picker.fig.savefig(os.path.join(wd,'lines.png'))
            p.export_vessel_diameters(name,save_figs_to=wd,format=self.diameter_save_format)
                
                
	
    def _load_btn_fired(self):
        
        if self.parent.frames is not None:
            del self.parent.frames
            self.parent.frames = None

        self.reset_fs()
        if self._verbose:
            print 'reset_fs done'

        self.dt,self.dtunits = self.fs.meta['axes'][0]
        if self._verbose:
            print self.fs.meta['axes']
            print 'dt done'

        self.parent._recalc_btn_fired()
        if self._verbose:
            print 'recalc'

class FrameViewer(HasTraits):
    fso = Instance(FrameSequenceOpts)
    figure = Instance(Figure, ())
    max_frames = Int(100)
    frame_index = Int(0)
    frames = None
    coords_stat = String()
    time_stat = String()
    recalc_btn = Button("Apply pipeline")

    frame_fwd_btn = Button('>')
    frame_back_btn = Button('<')


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
                title= "Image funcut UI",
                statusbar = [StatusItem('coords_stat'),
                             StatusItem('time_stat')])

    def _frame_fwd_btn_fired(self):
        self.frame_index += 1

    def _frame_back_btn_fired(self):
        self.frame_index -= 1

    def update_status_bar(self, event):
        if event.inaxes:
            x, y = event.xdata, event.ydata
            self.coords_stat = "x,y: %3.3f,%3.3f,"%(x,y)

    def _fso_default(self):
        fso = FrameSequenceOpts(self)
        return fso
    
    def _frame_index_changed(self):
        if hasattr(self, 'picker') and self.fso.fs is not None:
            dz,zunits = self.fso.fs.meta['axes'][0]
            t = self.frame_index * dz
            self.time_stat = "time: %3.2f, %s"%(t,zunits)
            self.picker.set_frame_index(self.frame_index)

    def redraw(self):
        try:
            self.pl.axes.figure.canvas.draw()
        except:
            print "Can't redraw"

    def _recalc_btn_fired(self):
        if len(self.figure.axes):
            self.axes = self.figure.axes[0]
            self.axes.clear()
        else:
            self.axes = self.figure.add_subplot(111)

        self.fso._fs2_needs_reload = True # enforce re-calculation
        fs2 = self.fso.get_fs()

        vl,vh = None, None

        if len(fs2.shape()) < 3:
            vl, vh = fs2.data_percentile((0.01, 99.99))
            print "vl,vh:", vl, vh
            self.fso.vmin, self.fso.vmax = float(vl), float(vh)

        Nf = len(fs2)
        if hasattr(self, 'picker'):
            self.picker.disconnect()
        self.picker = ifui.Picker(fs2)
	self.picker.caller = self

        ax1,self.pl,_ = self.picker.start(ax=self.axes, legend_type='axlegend',
                                          cmap = self.fso.colormap,
                                          vmin = vl, vmax=vh,
                                          interpolation = self.fso.interpolation)
        self.axes.figure.canvas.mpl_connect('motion_notify_event',
                                            self.update_status_bar)
        self.frame_index = 0
        self.max_frames = Nf-1
        wx.CallAfter(self.axes.figure.canvas.draw)


class _Camera(HasTraits):
    gain = Enum(1, 2, 3, )
    exposure = CInt(10, label="Exposure", )

class _DummySpatialFilterFn(HasTraits):
    name = 'Spatial filter'
    #active=Bool(False)
    sigma = Float(3.2)
    view = View('sigma')
    def __str__(self):
        return 'Dummy filter'
    
    
    

class TestApp(HasTraits):
    #ksf = Instance(KalmanStackFilterOpts)
    #cam = Instance(_Camera)
    gsf = Instance(_DummySpatialFilterFn)
    pipeline = List()
    add_btn = Button('Add')
    presets = {p.name:p for p in [KalmanStackFilter,_DummySpatialFilterFn]}
    preset_spin = Enum(*sorted(presets.keys()),label='inventory')
    view = View(Item('pipeline',
                     style='custom',
                     editor=ListEditor(use_notebook=True,
                                       page_name='.name',
                                       deletable=True)),
                HGroup('preset_spin',
                       'add_btn'),
                resizable=True)
    def _ksf_default(self):
        return KalmanStackFilterOpts()
    def _add_btn_fired(self):
        filt = self.presets[self.preset_spin]
        self.pipeline.append(filt())
        print self.pipeline

def main():
    FrameViewer().configure_traits()
    #TestApp(
    #    pipeline=[],
    #    ksf = KalmanStackFilterOpts(),
    #    gsf=_DummySpatialFilterFn()).configure_traits()

if __name__ == "__main__":
    main()


## class GWOpts(HasTraits):
##     pw_presets = {'4. identity' : lambda x,normL=110:x,
##                   '3. norm to SD' : lambda x, normL=110: x/np.std(x[:normL]),
##                   '2. DF/F' : lib.DFoF,
##                   '1. DF/SD' : lib.DFoSD,}
##     normL = Int(250, label="N baseline frames")
##     pw_func = Enum(*sorted(pw_presets.keys()), label='Pixel-wise')
##     tmedian_k = Enum([5]+range(1,13,2), label='median filter kernel')
##     gauss_size = Range(1.0, 20.0, 1.0, label='Gauss sigma (after)')
##     nclose = Range(0,5,1, 'iterations of 3D binary closing')
##     nopen = Range(0,5,1,  'iterations of 3D binary opening')
##     sigma_thresh = Range(0.0, 10.0, 1.5,
##                          label='Binarization threshold, x S.D.')
##     size_threshold = Range(1,2000,60,label='Volume threshold')
##     do_labels = Bool(True,info_text="Try to segment binary?")
##     run_btn = Button('Run')
##     roiexport_file = File('', label='Save ROIs to')
##     roiexport_btn = Button('Export ROIs')
##     view = View(Group(Item('normL'),
##                       Item('pw_func'),
##                       Item('tmedian_k'),
##                       Item('gauss_size'),
##                       show_border=True,
##                       label='Pre-process'),
##                 Group(Item('sigma_thresh'),
##                       Item('nopen'),
##                       Item('nclose'),
##                       show_border=True,
##                       label = 'Binarize',),
##                 Group(Item('size_threshold'),
##                       Item('do_labels', label='Do labels?'),
##                       show_border=True,
##                       label = 'Segment',),
##                 Item('run_btn', show_label=False),
##                 Group(Item('roiexport_file', label='Save rois to'),
##                       Item('roiexport_btn', show_label=False),
##                       show_border=True,
##                       label='Exporing'),
##                 )
##     def __init__(self, parent):
##         self.fso = parent
##     def _run_btn_fired(self):
##         if hasattr(self.fso, 'fs'):
##             seq = self.fso.fs
##             fn1 = partial(self.pw_presets[self.pw_func],
##                           normL=self.normL)
##             if self.tmedian_k < 3:
##                 pwfn = fn1
##             else:
##                 pwfn = lib.flcompose(fn1,
##                                       medfilt_fn(self.tmedian_k))
##             seq1 = seq.pw_transform(pwfn)
##             seq1.fns = [partial(filt.gauss_blur, size=self.gauss_size)]
##             arr = seq1.as3darray()     # new array with spatially-smoothed data
##             sds = float(np.std(arr))   # standard deviation in all data
##             binarr = arr > sds*self.sigma_thresh
##             if self.nopen > 0:
##                 binarr = ndimage.binary_opening(binarr, iterations=self.nopen)
##             if self.nclose >0:
##                 binarr = ndimage.binary_closing(binarr, iterations=self.nopen)
##             if self.do_labels:
##                 objects = find_objects(binarr,self.size_threshold)
##                 out = np.ma.array(objects, mask=objects==0)
##                 seq2 = fseq.FSeq_arr(out)
##                 cmap = 'hsv'
##             else:
##                 seq2 = fseq.FSeq_arr(binarr)
##                 cmap = 'jet'
##             self.fso.get_fs = lambda : seq2
##             self.fso.colormap = cmap
##             self.fso.parent._recalc_btn_fired()
##     def _roiexport_btn_fired(self):
##         print "!!!!"
##         picker = self.fso.parent.picker
##         #self.roiexport_file.configure_traits()
##         #picker.save_rois(self.roiexport_file)
##         f = file(self.roiexport_file,'w')
##         for c in self.fso.parent.picker.roi_objs.values():
##             if isinstance(c, ifui.CircleROI):
##                 f.write("%s: C=(%3.3f,%3.3f) R=%3.3f\n"%\
##                         (c.tag, c.obj.center[0], c.obj.center[1],
##                          c.obj.radius))
##         f.close()                    
##         pass
