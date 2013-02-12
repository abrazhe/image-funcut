### -------------------------------- ###
### Classes for sequences of frames ###
### -------------------------------- ###

import sys
import os
import re
import glob
import itertools as itt
import numpy as np
import tempfile as tmpf

_dtype_ = np.float64

from matplotlib.pyplot import imread

from imfun import lib
ifnot = lib.ifnot

def sorted_file_names(pattern):
    "Return a sorted list of file names matching a pattern"
    x = glob.glob(pattern)
    return sorted(x)

def iter_files(pattern, loadfn):
    """Return iterator over data frames, file names matching a pattern,
    loaded by a user-provided function loadfn
    """
    return itt.imap(loadfn, sorted_file_names(pattern))


def img_getter(frame, ch, pngflag):
    """A wrapper to extract color channel from image files.
    :returns: 2D matrix with intensity values
    """
    if len(frame.shape) > 2:
        out = frame[:,:,ch]
    else:
        out = frame
    if pngflag:
        out = out[::-1,:]
    return out

def _fseq_from_glob_deprecated(pattern, ch=None, loadfn=np.load):
    """Sequence of frames from filenames matching a glob
    
    :param pattern:  glob-pattern for file names
    :param ch: color channel
    :param loadfn : [np.load], function to load data from a file by its name

    :returns: iterator over 2D matrices
    """
    pngflag = (pattern[-3:] == 'png')
    return itt.imap(lambda frame: img_getter(frame, ch, pngflag),
                    iter_files(pattern, loadfn))


def fseq_from_glob(pattern, ch=None, loadfn=np.load):
    """Return sequence of frames from filenames matching a glob.

    Parameters:
      - pattern : (`str`) -- glob-style pattern for file names. 
      - ch: (`int` or `None`) -- color channel to extract if a number, all colors if `None`.
      - loadfn: (`func`) -- a function to load data from a file by its name [`np.load`].

    Returns:
      - iterator over frames. `2D` if `ch` is `int`, `3D` if ch is `None`
    """
    pngflag = (pattern[-3:] == 'png')
    if ch is not None:
	return itt.imap(lambda frame: img_getter(frame, ch, pngflag),
			iter_files(pattern, loadfn))
    else:
	return iter_files(pattern, loadfn)

class FrameSequence(object):
    "Base class for sequence of frames"
    def get_scale(self):
	"""
	Returns dx,dy and the flag whether the scale information has been set.
	If scale info hasn't been set, dx=dy=1
	"""
        if hasattr(self, '_scale_set'):
            scale_flag = self._scale_set
            dx,dy, scale_flag = self.dx, self.dy, self._scale_set
        else:
            dx,dy,scale_flag = 1,1,None
        return dx, dy, scale_flag

    def set_scale(self, dx=None, dy=None):
	"""sets self.dx, self.dy scale information"""
        self._scale_set = True
        if (dx is None) and (dy is None):
            self.dx,self.dy = 1,1
            self._scale_set = False
        elif (dx is not None) and (dy is None):
            self.dx, self.dy = dx, dx
        elif (dx is None) and (dy is not None):
            self.dx, self.dy = dy, dy
        else:
            self.dx, self.dy = dx,dy

    def std(self):
	"""get standard deviation of the data"""
        a = self.as3darray()
        return float(a.std())

    def data_range(self):
        """Return global range (`min`, `max`) values for the sequence"""
        ranges = np.array([(np.min(f), np.max(f)) for f in self.frames()])
        minv,maxv = np.min(ranges[:,0]), np.max(ranges[:,1])
        return (minv, maxv)

    def pipeline(self):
	"""Return the composite function to process frames based on self.fns"""
	return lib.flcompose(identity, *self.fns)

    def data_percentile(self, p):
	"""Return a percentile `p` value on data.

	Parameters:
	  - `p` : float in range of [0,100] (or sequence of floats)
	     Percentile to compute which must be between 0 and 100 inclusive.
	  
	"""
        return  np.percentile(self.as3darray(),p)

    def timevec(self,):
        """Return a vector of time stamps, calculated based on `self.dt`"""
        L = self.length()
        return np.arange(0, (L+2)*self.dt, self.dt)[:L]

    def mask_reduce(self, mask):
        """Return `1D` vector from a mask (or slice), taking average value within
        this mask in each frame"""
        return np.asarray([np.mean(f[mask]) for f in self.frames()])

    def frame_slices(self, sliceobj):
        """Return iterator over subframes (slices defined by `sliceobj` parameter).
	When `sliceobj` is `None`, full frames are returned 
	"""
        if sliceobj:
            return (f[sliceobj] for f in self.frames())
        else:
            return self.frames()

    def time_project(self,start=None,stop=None,fn=np.mean):
	"""Apply an ``f(vector) -> scalar`` function for each pixel.

	This is a more general (and often faster) function than
	`self.mean_frame` or `self.max_project`. However, it requires more
	memory as it makes a call to self.as3darray, while the other two don't
	
	Parameters:
	  - `start`: (`int` or `None`) -- frame to start at
	  - `stop`: (`int` or `None`) -- frame to stop at
	  - `fn` (`func`) -- function to apply (`np.mean` by default)

        Returns:
	  - `2D` `array` -- a projected frame
	"""
	L = self.length()
	start = ifnot(start,0)
        stop = min(ifnot(stop,L), L)
	out = np.zeros(self.shape())
	for v,r,c in self.pix_iter():
	    out[r,c] = fn(v[start:stop])
	return out

    def mean_frame(self, start=None, stop=None):
        """Return average image over N frames starting.

	Starts at `start`, stops at `stop`. A value of `None`
	corresponds to 0 for `start` and last frame for `stop`
	"""
        L = self.length()
        frameit = itt.imap(_dtype_, self.frames())
        res = np.copy(frameit.next())
	if start is None: start =0
        stop = min(ifnot(stop,L), L)
	count = 1
        for k,frame in enumerate(frameit):
	    if k < start: continue
	    elif k>=stop: break
            res += frame
	    count += 1
        return res/(count)
    
    def max_project(self, start = None, stop=None):
	"""Return max-projection image from start to stop frame
	(all by	default)
	"""
        L = self.length()
        frameit = itt.imap(_dtype_, self.frames())
	start = max(ifnot(start, 0), 0)
        stop = min(ifnot(stop,L), L)
	out = frameit.next()
        for k,frame in enumerate(frameit):
	    if k <= start: continue
	    elif k>stop: break
	    out = np.max([out, frame], axis=0)
        return out

    def aslist(self, max_frames=None, sliceobj=None):
        """Return the frames as a list up to `max_frames` frames
	taking ``f[sliceobj]`` slices.
	"""
        return list(self.asiter(max_frames,sliceobj))

    def asiter(self, max_frames=None, sliceobj=None):
        """Return an iterator over the up to `max_frames` frames taking
	``f[sliceobj]`` slices
	"""
        fiter = self.frame_slices(sliceobj)
        return itt.islice(fiter, max_frames)

    def as3darray(self,  max_frames=None,sliceobj=None,
                  dtype = _dtype_):
	"""Return the frames as a `3D` array.

	An alternative way is to use the __get_item__ interface:
	``data = np.asarray(fs[10:100])``
	

	Parameters:
	  - `max_frames`: (`int` or `None`) -- up to this frame. Up to last frame if
	    `None`
	  - `sliceobj`: (`slice` or `None`) -- a slice to take from each frame
	  - `dtype`: (`type`) -- data type to use. Default, ``np.float64``

	Returns:
	  `3D` array `d`, where frames are stored in higher dimensions, such
	  that ``d[0]`` is the first frame, etc.
	"""
        fiter = self.frame_slices(sliceobj)
        shape = self.shape(sliceobj)
	newshape = [self.length()] + list(shape)
        out = lib.memsafe_arr(newshape, dtype)
        for k,frame in enumerate(itt.islice(fiter, max_frames)):
            out[k,:,:] = frame
        if hasattr (out, 'flush'):
            out.flush()
        return out
    
    def pix_iter(self, mask=None, max_frames=None, rand=False, **kwargs):
        """Return iterator over time signals from each pixel.

	Parameters:
	  - `mask`: (2D `Bool` array or `None`) -- skip pixels where `mask` is
            `False` if `mask` is `None`, take all pixels
	  - `max_frames`: (`int` or `None`) -- use frames up to `max_frames`
	  - `rand`: (`Bool`) -- whether to go through pixels in a random order
	  - `**kwargs`: keyword arguments to be passed to `self.as3darray`

	Yields:
	 tuples of `(v,row,col)`, where `v` is the time-series in a pixel at `row,col`
	
	"""
        arr = self.as3darray(max_frames, **kwargs)
        if mask== None:
            mask = np.ones(self.shape(), np.bool)
        nrows, ncols = arr.shape[1:]
        rcpairs = [(r,c) for r in xrange(nrows) for c in xrange(ncols)]
        if rand: rcpairs = np.random.permutation(rcpairs)
        for row,col in rcpairs:
            if mask[row,col]:
                ## asarray to convert from memory-mapped array
                yield np.asarray(arr[:,row,col]), row, col
        if hasattr(arr, 'flush'):
            arr.flush()
        del arr
        

    def length(self):
        """Return number of frames in the sequence"""
        if not hasattr(self,'_length'):
            k = 0
            for _ in self.frames():
                k+=1
            self._length = k
            return k
        else:
            return self._length

    def shape(self, sliceobj=None):
        "Return the shape of frames in the sequence"
        return self.frame_slices(sliceobj).next().shape

    def _norm_mavg(self, tau=90., **kwargs):
	"Return normalized frame sequence"
	from scipy import ndimage
	if kwargs.has_key('dtype'):
	    dtype = kwargs['dtype']
	else:
	    dtype = _dtype_
	dt = self.dt
	arr = self.as3darray(**kwargs)
	sigma = tau/dt
	smooth =  ndimage.gaussian_filter1d(arr, sigma, axis=0)
	zi = np.where(np.abs(smooth) < 1e-6)
	out  = arr/smooth - 1.0
	out[zi] = 0
	return FSeq_arr(out, dt = dt, dx=self.dx, dy = self.dy)
	

    def pw_transform(self, pwfn, verbose=False, **kwargs):
        """Spawn another frame sequence, pixelwise applying a user-provided
        function.

	Parameters:
	  - `pwfn`: (`func`) -- a ``f(vector) -> vector`` function
	  - `verbose`: (`Bool`) -- whether to be verbose while going through
            the pixels
	  - `**kwargs``: keyword arguments to be passed to `self.pix_iter`
	"""
	nrows, ncols = self.shape()[:2]
	if kwargs.has_key('dtype'):
	    dtype = kwargs['dtype']
	else:
	    dtype = _dtype_
	L = len(pwfn(np.random.randn(self.length())))
	#testv = pwfn(self.pix_iter(rand=True,**kwargs).next()[0])
	#L = len(testv)
	out = lib.memsafe_arr((L, nrows, ncols), dtype)
        for v, row, col in self.pix_iter(**kwargs):
	    if verbose:
		sys.stderr.write('\rworking on pixel (%03d,%03d)'%(row, col))
            out[:,row,col] = pwfn(v)
	    if hasattr(out, 'flush'):
		out.flush()
	dt = self.dt*self.length()/L
        return FSeq_arr(out, dt = dt, dx=self.dx, dy = self.dy)

    def export_img(self, path, base = 'fseq-export-', figsize=(4,4),
                   start = 0, stop = None, show_title = True,
                   format='.png', vmin = None, vmax=None, **kwargs):
	"""Export frames as images by drawing them with ``pylab.imshow``.

	Parameters:
	  - `path` : (`str`) -- directory where to save images to. Will be created
	    if doesn't exist
	  - `base` : (`str`) -- a prefix for the created file names
	  - `figsize`: (`tuple` or `array-like`) -- size of the figures in inches
	  - `start` : (`int`) -- start at this frame
	  - `stop` : (`int` or `None`) -- stop at this frame
	  - `show_title`: (`Bool`) -- flag whether to show a title over the
	    frame
	  - `format`: (`str`) -- output format, can be png, svg, eps, pdf,
	    bmp,tif
	  - `vmin` : (`number` or `None`) -- to be passed to imshow. If `None`,
	    global minimum over the frame sequence is used.
	  - `vmax` : (`number` or `None`) -- to be passed to imshow. If `None`,
	    global maximum over the frame sequence is used.
	  - `**kwargs`: other arguments that will be passed to `imshow`

	Returns:
	  - a list of names for the created files
	
	"""
        import  sys
        import matplotlib.pyplot as plt
        lib.ensure_dir(path)
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        if stop is None or stop == -1:
            stop = self.length()
	if hasattr(self, 'data'):
	    vmin = ifnot(vmin, self.data_percentile(1)) # for scale
	    vmax = ifnot(vmax, self.data_percentile(99)) # for scale
	else:
	    vmin = ifnot(vmin, np.min(map(np.min, self.frames())))
	    vmax = ifnot(vmax, np.min(map(np.max, self.frames())))
        kwargs.update({'vmin':vmin, 'vmax':vmax})
	print path+base
        L = min(stop-start, self.length())
	fnames = []
        for i,frame in enumerate(self.frames()):
            if i < start: continue
            if i > stop: break
            ax.cla()
            ax.imshow(frame, aspect='equal', **kwargs)
            fname =  path + base + '%06d.png'%i
	    fnames.append(fname)
            if show_title:
                ax.set_title('frame %06d (%3.3f s)'%(i, i*self.dt))
            fig.savefig(fname)
            sys.stderr.write('\r saving frame %06d of %06d'%(i+1, L))
        plt.close()
	return fnames
    def export_mpeg(self, mpeg_name, fps = None, **kwargs):
	"""Create an mpg  movie from the frame sequence using mencoder.

	Parameters:
	  - `mpeg_name`: (`str`) -- a name (without extension) for the movie to
	    be created
	  - `fps`: (`number`) -- frames per second. If None, use 10/self.dt
	  - `**kwargs` : keyword arguments to be passed to `self.export_png`
	"""
        print "Saving frames as png"
        if fps is None:
            fps = 10/self.dt
	if not kwargs.has_key('path'):
	    kwargs['path'] = './'
	if not kwargs.has_key('base'):
	    kwargs['base'] = '-mpeg-export'
	path,base = kwargs['path'], kwargs['base']
	fnames = self.export_img(**kwargs)
        print 'Running mencoder, this can take a while'
        mencoder_string = """mencoder mf://%s*.png -mf type=png:fps=%d\
        -ovc lavc -lavcopts vcodec=wmv2 -oac copy -o %s.mpg"""%(path+base,fps,mpeg_name)
        os.system(mencoder_string)
        map(os.remove, fnames)

class FSeq_arr(FrameSequence):
    """A FrameSequence class as a wrapper around a `3D` array
    """
    def __init__(self, arr, dt = 1.0, fns = [],
                 dx = None, dy = None):
        self.dt = dt
        self.data = arr
        self.hooks = []
        self.fns = fns
        self.set_scale(dx, dy)
    def length(self):
        return self.data.shape[0]
    def __getitem__(self, val):
	x = self.data[val]
	if self.fns == []:
	    return self.data[val]
	if type(val) is int:
	    return self.pipeline()(x)
	out = np.zeros(x.shape,x.dtype)
	for j,f in enumerate(x):
	    out[j] = self.pipeline()(x)
	return out
    def data_percentile(self, p):
        #from scipy.stats import scoreatpercentile
        return np.percentile(np.ravel(self.data),p)

    def pix_iter(self, mask=None,max_frames=None,rand=False,**kwargs):
	"Iterator over time signals from each pixel (FSeq_arr)"
	if self.fns == []:
	    if mask == None:
		mask = np.ones(self.shape(), np.bool)
	    nrows, ncols = self.shape()
	    if kwargs.has_key('dtype'):
		dtype = kwargs['dtype']
	    else: dtype=_dtype_
	    rcpairs = [(r,c) for r in xrange(nrows) for c in xrange(ncols)]
	    if rand: rcpairs = np.random.permutation(rcpairs)
	    for row,col in rcpairs:
		if mask[row,col]:
		    v = self.data[:,row,col].copy()
		    yield np.asarray(v, dtype=dtype), row, col
	else:
	    x = super(FSeq_mlf,self)
	    for a in x.pix_iter(mask=mask,max_frames=max_frames,
				rand=rand,**kwargs):
		yield a


    def frames(self):
	"""
	Return iterator over frames.

	The composition of functions in `self.fns` list is applied to each
	frame. By default, this list is empty.  Examples of function"hooks"
	to put into `self.fns` are ``imfun.lib.DFoSD``,
	``imfun.lib.DFoF`` or functions from ``scipy.ndimage``.
	"""
        fn = self.pipeline()
        return itt.imap(fn, (frame for frame in self.data))


def identity(x):
    return x

class FSeq_glob(FrameSequence):
    """A FrameSequence class as a wrapper around a set of files matching a
    glob-like pattern"""
    def __init__(self, pattern, ch=0, dt = 1.0, fns = [],
                 dx = None, dy = None):
        self.pattern = pattern
        self.ch = ch
        self.dt = dt
        self.fns = fns
        self.set_scale(dx, dy)
	self.file_names = sorted_file_names(pattern)
    def length(self):
	return len(self.file_names)
            
    def frames(self, ch = None):
	"""
	Return iterator over frames.

	The composition of functions in `self.fns`
	list is applied to each frame. By default, this list is empty. Examples
	of function "hooks" to put into `self.fns` are ``imfun.lib.DFoSD``,
	``imfun.lib.DFoF`` or functions from ``scipy.ndimage``.
	"""
        fn = self.pipeline()
        return itt.imap(fn, fseq_from_glob(self.pattern,
					   ifnot(ch, self.ch), self.loadfn))
    def __getitem__(self, val):
	pngflag = self.file_names[0][-3:] == 'png'
	fn = self.pipeline()
	if type(val) is int:
	    frame = self.loadfn(self.file_names[val])
	    if self.ch is not None:
		frame = img_getter(frame, self.ch, pngflag)
	    return fn(frame)
	else:
	    seq =  map(self.loadfn, self.file_names[val])
	    if self.ch is not None:
		seq = (img_getter(f, self.ch, pngflag) for f in seq)
	    return map(fn, seq)
	
class FSeq_img(FSeq_glob):
    """FrameSequence around a set of image files"""
    loadfn = lambda self,y: imread(y)

class FSeq_txt(FSeq_glob):
    """FrameSequence around a set of text-image files"""
    loadfn= lambda self,y: np.loadtxt(y)

class FSeq_npy(FSeq_glob):
    """FrameSequence around a set of npy files"""
    loadfn= lambda self,y: np.load(y)

class FSeq_imgleic(FSeq_img):
    """FrameSequence around the image files created by LeicaSoftware.
    It is just a wrapper around FSeq_img, only it also looks for an xml
    file in Leica's format with the Job description
    """
    def __init__(self, pattern, ch=0, fns=[], xmlname = None,
                 dt = 1.0, dx = None, dy = None):
        FSeq_glob.__init__(self, pattern,ch=ch)
        if xmlname is None:
            xmlname = self.pattern.split('*')[0]
        self.fns = fns
        try:
            from imfun import leica
            self.lp = leica.LeicaProps(xmlname)
            self.dt = self.lp.dt # sec
            self.set_scale(self.lp.dx, self.lp.dy) # um/pix
        except Exception as e:
            print "Got exception, ", e
            # Set fallback options
            self.dt = dt
            self.set_scale(dx,dy) # um/pix
            pass


#from imfun.MLFImage import MLF_Image
from imfun import MLFImage

class FSeq_mlf(FrameSequence):
    "FrameSequence class for MLF multi-frame images"
    def __init__(self, fname, fns = []):
        self.mlfimg = MLFImage.MLF_Image(fname)
        self.dt = self.mlfimg.dt/1000.0
        self.fns = []
        self.set_scale()
    def frames(self,):
	"""
	Return iterator over frames.

	The composition of functions in `self.fns`
	list is applied to each frame. By default, this list is empty. Examples
	of function "hooks" to put into `self.fns` are ``imfun.lib.DFoSD``,
	``imfun.lib.DFoF`` or functions from ``scipy.ndimage``.
	"""
	
        fn = lib.flcompose(identity, *self.fns)
        return itt.imap(fn,self.mlfimg.flux_frame_iter())
    def __getitem__(self, val):
	L = self.length()
	fn = self.pipeline()
	if type(val) is int:
	    return fn(self.mlfimg[val])
	else:
	    indices = range(self.mlfimg.nframes)
	    return itt.imap(fn, itt.imap(self.mlfimg.read_frame, indices[val]))
    def length(self):
        return self.mlfimg.nframes
    def pix_iter(self, mask=None, max_frames=None, rand=False, **kwargs):
        "Iterator over time signals from each pixel"
	if self.fns == []:
	    if mask == None:
		mask = np.ones(self.shape(), np.bool)
	    nrows, ncols = self.shape()
	    if kwargs.has_key('dtype'):
		dtype = kwargs['dtype']
	    else: dtype=_dtype_
	    rcpairs = [(r,c) for r in xrange(nrows) for c in xrange(ncols)]
	    if rand: rcpairs = np.random.permutation(rcpairs)
	    for row,col in rcpairs:
		if mask[row,col]:
		    #v = [f[row,col] for f in self[:max_frames]]
		    v = self.mlfimg.read_timeslice((row,col))
		    yield np.asarray(v, dtype=dtype), row, col
	else:
	    x = super(FSeq_mlf,self)
	    for a in x.pix_iter(mask=mask,max_frames=max_frames,
				rand=rand,**kwargs):
		yield a
		
	    #FrameSequence.pix_iter(self, mask=mask,max_frames=max_frames,rand=rand,**kwargs)

import PIL.Image as Image
import matplotlib.image as mpl_img
class FSeq_multiff(FrameSequence):
    "Class for multi-frame tiff files"
    def __init__(self, fname, dt=1.0):
        self.dt = dt
        self.fns = []
        self.im = Image.open(fname)
        self.set_scale()
    def frames(self, count=0):
	"""
	Return iterator over frames.

	The composition of functions in `self.fns`
	list is applied to each frame. By default, this list is empty. Examples
	of function "hooks" to put into `self.fns` are functions from ``scipy.ndimage``.
	"""
        fn = self.pipeline()
        while True:
            try:
                self.im.seek(count)
                count += 1
                yield fn(mpl_img.pil_to_array(self.im))
            except EOFError:
                break
            
def open_seq(path, *args, **kwargs):
    """Dispatch to an appropriate class constructor depending on the file name

    Parameters:
      - path: (`str`) -- path to load data from. Can be a glob-style pattern or
        a single file name.
      - `*args`, `**kwargs`: will be dispatched to the actual class' `__init__` call

    Returns:
      - `instance`  of an appropriate Frame Sequence class
    """
    images =  ('bmp', 'jpg', 'jpeg', 'png', 'tif','tiff')
    if type(path) is np.ndarray:
	return FSeq_arr(path, *args, **kwargs)
    ending = re.findall('[^*\.]+', path)[-1]
    if ending == 'txt':
        return FSeq_txt(path, *args, **kwargs)
    elif ending == 'mlf':
        return FSeq_mlf(path, *args, **kwargs)
    elif ending == 'npy':
        return FSeq_npy(path, *args, **kwargs)
    elif ending in images:  # A collection of images or a big tiff
        if '*' in path: # many files
            from imfun import leica
            xml_try = leica.get_xmljob(path.split('*')[0])
            if kwargs.has_key('xmlname') or xml_try:
                return FSeq_imgleic(path, *args, **kwargs)
            else:
                return FSeq_img(path, *args, **kwargs)
        elif ending in ('tif', 'tiff'): # single multi-frame tiff
            return FSeq_multiff(path, *args, **kwargs)
            
    
