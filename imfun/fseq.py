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


def img_getter(frame, ch):
    """A wrapper to extract color channel from image files.
    :returns: 2D matrix with intensity values
    """
    if len(frame.shape) > 2:
        out = frame[:,:,ch]
    else:
        out = frame
    return out


def fseq_from_glob(pattern, ch=None, loadfn=np.load):
    """Return sequence of frames from filenames matching a glob.

    Parameters:
      - pattern : (`str`) -- glob-style pattern for file names. 
      - ch: (`int` or `None`) -- color channel to extract if a number, all colors if `None`.
      - loadfn: (`func`) -- a function to load data from a file by its name [`np.load`].

    Returns:
      - iterator over frames. `2D` if `ch` is `int`, `3D` if ch is `None`
    """
    if ch is not None:
	return itt.imap(lambda frame: img_getter(frame, ch),
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

    def pipeline(self):
	"""Return the composite function to process frames based on self.fns"""
	return lib.flcompose(identity, *self.fns)

    def std(self):
	"""get standard deviation of the data"""
        a = self.as3darray()
        return float(a.std())

    def data_range(self):
        """Return global range (`min`, `max`) values for the sequence"""
        # need this strange syntax for min/max for multi-channel images
        rfn = lambda fn: lambda x: fn(fn(x, axis=0),axis=0)
        ranges = np.array([(rfn(np.min)(f), rfn(np.max)(f)) for f in self.frames()])
        minv,maxv = np.min(ranges[:,0],axis=0), np.max(ranges[:,1],axis=0)
        return np.array([minv,maxv]).T
        #return (minv, maxv)

    def data_percentile(self, p):
	"""Return a percentile `p` value on data.

	Parameters:
	  - `p` : float in range of [0,100] (or sequence of floats)
	     Percentile to compute which must be between 0 and 100 inclusive.
	  
	"""
        sh = self.shape()
        arr = self.as3darray()
        if len(sh) == 2:
            return  np.percentile(arr,p)
        else:
            return [np.percentile(arr[...,k],p) for k in range(sh[2])]

    def timevec(self,):
        """Return a vector of time stamps, calculated based on `self.dt`"""
        L = self.length()
        return np.arange(0, (L+2)*self.dt, self.dt)[:L]

    def mask_reduce(self, mask):
        """Return `1D` vector from a mask (or slice), taking average value within
        this mask in each frame"""
        return np.asarray([np.mean(f[mask],axis=0) for f in self.frames()])
    def softmask_reduce(self,mask):
	"""Return `1D` vector from a mask (or slice), taking average value within
        this mask in each frame, weighted by mask values between 0 and 1"""
        sh = self.shape()
        mask2d = mask
        if len(sh) >2 :
            mask = np.dstack([mask]*sh[-1])
        return np.asarray([np.mean((f*mask)[mask2d>0],axis=0) for f in self.frames()])


    def frame_slices(self, crop):
        """Return iterator over subframes (slices defined by `crop` parameter).
	When `crop` is `None`, full frames are returned 
	"""
        if crop:
            return (f[crop] for f in self.frames())
        else:
            return self.frames()

    def time_project(self,fslice=None,fn=np.mean,crop=None):
	"""Apply an ``f(vector) -> scalar`` function for each pixel.

	This is a more general (and often faster) function than
	`self.mean_frame` or `self.max_project`. However, it requires more
	memory as it makes a call to self.as3darray, while the other two don't
	
	Parameters:
	  - `fslice`: (`int`, `slice` or `None`) -- go throug these frames
	  - `fn` (`func`) -- function to apply (`np.mean` by default)

        Returns:
	  - `2D` `array` -- a projected frame
	"""
        sh = self.shape(crop)
	out = np.zeros(sh)
        if len(sh)>2:
            fn = lambda a: np.mean(a, axis=0)
	for v,r,c in self.pix_iter(fslice=fslice,crop=crop):
	    out[r,c] = fn(v)
	return out

    def mean_frame(self, fslice=None):
        """Return average image over N frames starting.

	Starts at `start`, stops at `stop`. A value of `None`
	corresponds to 0 for `start` and last frame for `stop`
	"""
        if fslice is None or type(fslice) is int:
            fslice = (fslice, )
        frameit = itt.imap(_dtype_, itt.islice(self.frames(), *fslice))
        res = np.copy(frameit.next())
        count = 0
        for k,frame in enumerate(frameit):
            res += frame
	    count += 1
        return res/(count)
    
    def max_project(self, fslice):
	"""Return max-projection image from start to stop frame
	(all by	default)
	"""
        if fslice is None or type(fslice) is int:
            fslice = (fslice, )
	out = frameit.next()
        frameit = itt.imap(_dtype_, itt.islice(self.frames(), *fslice))
        for k,frame in enumerate(frameit):
	    if k <= start: continue
	    elif k>stop: break
	    out = np.max([out, frame], axis=0)
        return out

    def aslist(self, max_frames=None, crop=None):
        """Return the frames as a list up to `max_frames` frames
	taking ``f[crop]`` slices.
	"""
        print "fseq.aslist is deprecated and will be removed\
        use the  __getitem__ interface, e.g. fseq[10:20]"
        return list(self.asiter(max_frames,crop))

    def asiter(self, fslice=None, crop=None):
        """Return an iterator over the frames taking frames from fslice and
	``f[crop]`` for each frame 
	"""
        fiter = self.frame_slices(crop)
        if type(fslice) is int :
            fslice = (fslice, )
        return itt.islice(fiter, *fslice)

    def as3darray(self,  fslice=None, crop=None,
                  dtype = _dtype_):
	"""Return the frames as a `3D` array.

	//An alternative way is to use the __get_item__ interface:
	//``data = np.asarray(fs[10:100])``
	

	Parameters:
	  - `fslice`: (`int`, `slice` or `None`) -- slice to go through frames
	  - `crop`: (`slice` or `None`) -- a crop (tuple of slices) to take from each frame
	  - `dtype`: (`type`) -- data type to use. Default, ``np.float64``

	Returns:
	  `3D` array `d`, where frames are stored in higher dimensions, such
	  that ``d[0]`` is the first frame, etc.
	"""
        if fslice is None or type(fslice) is int:
            fslice = (fslice, )
        shape = self.shape(crop)
	newshape = (self.length(),) + shape
        out = lib.memsafe_arr(newshape, dtype)
        for k,frame in enumerate(itt.islice(self.frames(), *fslice)):
            out[k,:,:] = frame[crop]
        out = out[:k+1]
        if hasattr (out, 'flush'):
            out.flush()
        return out
    
    def pix_iter(self, pmask=None, fslice=None, rand=False,
		 crop = None,
		 **kwargs):
        """Return iterator over time signals from each pixel.

	Parameters:
	  - `mask`: (2D `Bool` array or `None`) -- skip pixels where `mask` is
            `False` if `mask` is `None`, take all pixels
          - `fslice`: (`int`, `slice` or `None`) -- slice to go through frames
	  - `rand`: (`Bool`) -- whether to go through pixels in a random order
	  - `**kwargs`: keyword arguments to be passed to `self.as3darray`

	Yields:
	 tuples of `(v,row,col)`, where `v` is the time-series in a pixel at `row,col`
	
	"""
        arr = self.as3darray(fslice, crop=crop,**kwargs)
	sh = self.shape(crop)
        if pmask== None:
            pmask = np.ones(sh[:2], np.bool)
        nrows, ncols = sh[:2]
        rcpairs = [(r,c) for r in xrange(nrows) for c in xrange(ncols)]
        if rand: rcpairs = np.random.permutation(rcpairs)
	if crop is None:
	    submask = pmask
	    r0,c0 = 0,0
	else:
	    submask = pmask[crop]
	    r0,c0 = crop[0].start,crop[1].start
        for row,col in rcpairs:
	    if r0+row>=sh[0] or c0+col>=sh[1]:
		continue
            if submask[row,col]:
                ## asarray to convert from memory-mapped array
                yield np.asarray(arr[:,row,col]), row, col
        if hasattr(arr, 'flush'):
            arr.flush()
        del arr
	return
        

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

    def shape(self, crop=None):
        "Return the shape of frames in the sequence"
        return self.frame_slices(crop).next().shape

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
	#nrows, ncols = self.shape()[:2]
	if kwargs.has_key('dtype'):
	    dtype = kwargs['dtype']
	else:
	    dtype = _dtype_
	L = len(pwfn(np.random.randn(self.length())))
	#testv = pwfn(self.pix_iter(rand=True,**kwargs).next()[0])
	#L = len(testv)
	out = lib.memsafe_arr((L,) + self.shape(), dtype)
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
    
    def export_movie_anim(self, mpeg_name, fps=None, start=0, stop=None,
                          show_title=True, fig_size=(4,4),
                          vmin=None, vmax=None,**kwargs):
        """
        Create an mpg  movie from the frame sequence using mencoder.
        and mpl.Animation

	Parameters:
	  - `mpeg_name`: (`str`) -- a name (without extension) for the movie to
	    be created
	  - `fps`: (`number`) -- frames per second. If None, use 10/self.dt
	  - `**kwargs` : keyword arguments to be passed to `self.export_png`
	"""
        from matplotlib import animation
        import matplotlib.pyplot as plt

        if fps is None:
            fps = 0.5/self.dt

        if stop is None or stop == -1:
            stop = self.length()
	if hasattr(self, 'data'):
	    vmin = ifnot(vmin, self.data_percentile(1)) # for scale
	    vmax = ifnot(vmax, self.data_percentile(99)) # for scale
	else:
	    vmin = ifnot(vmin, np.min(map(np.min, self.frames())))
	    vmax = ifnot(vmax, np.min(map(np.max, self.frames())))
        kwargs.update({'vmin':vmin, 'vmax':vmax})
        L = min(stop-start, self.length())

        fig = plt.figure(figsize=fig_size)
        ax = fig.add_subplot(111)
        plh = ax.imshow(self[start], 
                        aspect='equal', **kwargs)
        mytitle = ax.set_title('')
        def _init():
            k = 0
            if show_title:
                mytitle.set_text('frame: %04d, time: %0.3f s'%(k, k*self.dt))
            plh.set_data(self[k])
            return plh, 
        def _animate(framecount):
            k = framecount+start
            plh.set_data(self[k])
            if show_title:
                mytitle.set_text('frame: %04d, time: %0.3f s'%(k, k*self.dt))
            return plh,
        
        anim = animation.FuncAnimation(fig, _animate, init_func=_init, frames=L, blit=True)
        mencoder_extra_args=['-ovc', 'lavc', '-lavcopts', 'vcodec=mpeg4']
        plt.close(anim._fig)
        anim.save(mpeg_name, writer='mencoder', fps=fps, extra_args=mencoder_extra_args)
        return 

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
        sh = self.shape()
        arr = self.data
        if len(sh) == 2:
            return  np.percentile(arr,p)
        else:
            return [np.percentile(arr[...,k],p) for k in range(sh[2])]

    def pix_iter(self, pmask=None,fslice=None,rand=False,**kwargs):
	"Iterator over time signals from each pixel (FSeq_arr)"
	if self.fns == []:
	    if pmask == None:
                pmask = np.ones(self.shape()[:2], np.bool)
	    nrows, ncols = self.shape()[:2]
	    if kwargs.has_key('dtype'):
		dtype = kwargs['dtype']
	    else: dtype=_dtype_
	    rcpairs = [(r,c) for r in xrange(nrows) for c in xrange(ncols)]
	    if rand: rcpairs = np.random.permutation(rcpairs)
	    for row,col in rcpairs:
		if pmask[row,col]:
		    v = self.data[:,row,col].copy()
		    yield np.asarray(v, dtype=dtype), row, col
	else:
	    x = super(FSeq_arr,self)
	    for a in x.pix_iter(pmask=pmask,fslice=fslice,
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
	fn = self.pipeline()
	if type(val) is int:
	    frame = self.loadfn(self.file_names[val])
	    if self.ch is not None:
		frame = img_getter(frame, self.ch)
	    return fn(frame)
	else:
	    seq =  map(self.loadfn, self.file_names[val])
	    if self.ch is not None:
		seq = (img_getter(f, self.ch) for f in seq)
            return map(fn, seq)
            
class FSeq_img(FSeq_glob):
    """FrameSequence around a set of image files"""
    loadfn = lambda self,y: imread(y)

class FSeq_txt(FSeq_glob):
    """FrameSequence around a set of text-image files"""
    loadfn= lambda self,y: np.loadtxt(y)

## TODO: but npy can be just one array
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
    def pix_iter(self, pmask=None, fslice=None, rand=False, **kwargs):
        "Iterator over time signals from each pixel, where pmask[pixel] is True"
	if self.fns == []:
	    if pmask == None:
		pmask = np.ones(self.shape(), np.bool)
	    nrows, ncols = self.shape()
	    if kwargs.has_key('dtype'):
		dtype = kwargs['dtype']
	    else: dtype=_dtype_
	    rcpairs = [(r,c) for r in xrange(nrows) for c in xrange(ncols)]
	    if rand: rcpairs = np.random.permutation(rcpairs)
	    for row,col in rcpairs:
		if pmask[row,col]:
		    v = self.mlfimg.read_timeslice((row,col))
		    yield np.asarray(v, dtype=dtype), row, col
	else:
	    x = super(FSeq_mlf,self)
	    for a in x.pix_iter(pmask=pmask,fslice=fslice,
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
class FSeq_tiff_2(FSeq_arr):
    "Class for (multi-frame) tiff files, using tiffile.py by Christoph Gohlke"
    def __init__(self, fname, ch=None, flipv = False, fliph = False, **kwargs):
	import tiffile
	x = tiffile.imread(fname)
	parent = super(FSeq_tiff_2, self)
	parent.__init__(x, **kwargs)
        if ch is not None and self.data.ndim > 3:
            self.data = self.data[:,:,:,ch]
        if flipv:
            self.data = self.data[:,::-1,...]
        if fliph:
            self.data = self.data[:,:,::-1,...]



## -- MES files --
## TODO: move to a separate file?
    
import mesa                      

class FSeq_mes(FSeq_arr):
    def __init__(self, fname, record=1, ch=None, fns=[],verbose=False):
        """
        The following format is assumed:
        the matlab (.mes) file contains description in a field like "Df0001",
        and actual data records in fields like 'If0001_001', 'If0001_002'.
        These fields contain data for the red and green channel, accordingly
        The images are stored as NXM arrays, where N is one side of an image,
        and then columns iterate over the other image dimension and time.
        """
        self.ch = ch
        self.fns = fns
        self.file_name = fname
        self.record = record
        self._verbose=verbose

        if type(record) is int:
            record = 'Df%04d'%record
        elif type(record) is str:
            if not ('Df' in record):
                record = 'Df%04d'%int(record)
        else:
            print "Unknown record definition format"

        meta = mesa.load_meta(fname)
        if verbose:
            print "file %s has following records:"%fname
            #print record_keys(meta)
            for k in mesa.record_keys(meta):
                if mesa.is_zstack(meta[k]):
                    desc = 'z_stack'
                elif mesa.is_xyt(meta[k]):
                    desc = 'XYT measure'
                else:
                    desc = 'unknown'
                print k, 'is a', desc

        entry = meta[record]
        if not mesa.is_xyt(entry):
            print "record %s is not an XYT measurement"%record
            return

        self._rec_meta = entry
        self.date = mesa.get_date(entry)
        self.ffi = mesa.get_ffi(entry)

        nframes, sh = mesa.get_xyt_shape(entry)
        self._nframes = nframes
        self._nlines = sh[1]
        self._shape = sh
        self.dt = mesa.get_sampling_interval(self.ffi)
        self.dx = self.dy = 1 #TODO: fix this
        streams = self.load_record(record)
        base_shape = (self._nframes-1, self._nlines, self._linesize)
        if ch is not None:
            stream = streams[ch]
            self.data = np.zeros(base_shape, dtype=stream.dtype)
            for k,f in enumerate(self._reshape_frames(stream)):
                self.data[k] = f
        else:
            streams = [lib.clip_and_rescale(s) for s in streams]
            reshape_iter = itt.izip(*map(self._reshape_frames, streams))
            sh = base_shape + (max(3, len(streams)),)
            self.data = np.zeros(sh, dtype=streams[0].dtype)
            for k, a in enumerate(reshape_iter):
                for j, f in enumerate(a):
                    self.data[k,...,j] = f
    

    def load_record(self, record):
        meas_info = mesa.only_measures(self._rec_meta)
        var_names = [x[0] for x in meas_info['ImageName']]
        var_names.sort()
        recs = mesa.io.loadmat(self.file_name, variable_names=var_names)
        streams = [recs[n] for n in var_names if n in recs]
        if len(streams) == 0:
            raise IndexError("can't load record number %d"%record)
        self._linesize = streams[0].shape[0]
        if self._verbose:
            print 'Number of working channels:', len(streams)
        return streams

    def _reshape_frames(self, stream):
        side = self._nlines
        return (stream[:,k*side:(k+1)*side].T for k in xrange(1,self._nframes))
      


## -- End of MES files --        


import inspect

def open_seq(path, *args, **kwargs):
    """Dispatch to an appropriate class constructor depending on the file name

    Parameters:
      - path: (`str`) -- path to load data from. Can be a glob-style pattern or
        a single file name.
      - `*args`, `**kwargs`: will be dispatched to the actual class' `__init__` call

    Returns:
      - `instance`  of an appropriate Frame Sequence class
    """
    images =  ('bmp', 'jpg', 'jpeg', 'png', 'tif','tiff', 'ppm', 'pgm')
    if type(path) is np.ndarray:
	return FSeq_arr(path, *args, **kwargs)
    ending = re.findall('[^*\.]+', path)[-1].lower()
    if ending == 'txt':
        handler = FSeq_txt
    elif ending == 'mes':
        handler = FSeq_mes
    elif ending == 'mlf':
        handler = FSeq_mlf
    elif ending == 'npy':
        handler =  FSeq_npy
    elif ending in images:  # A collection of images or a big tiff
        if '*' in path: # many files
            from imfun import leica
            xml_try = leica.get_xmljob(path.split('*')[0])
            if kwargs.has_key('xmlname') or xml_try:
                handler =  FSeq_imgleic
            else:
                handler =  FSeq_img
        elif ending in ('tif', 'tiff'): # single multi-frame tiff
            handler = FSeq_tiff_2
    spec = inspect.getargspec(handler.__init__)
    for k in kwargs.keys():
        if k not in spec[0]:
            kwargs.pop(k) # skip arguments the __init__ doesn't know about
    return handler(path, *args, **kwargs)



    
## HDF5-related stuff

try:
    import h5py

    class FSeq_hdf5(FSeq_arr):
        "Base class for hdf5 files"
        def __init__(self, fname, dataset=None,**kwargs):
            parent = super(FSeq_hdf5, self)
            f = h5py.File(fname, 'r')

            if dataset and dataset not in f:
                print "Dataset name doesn't exist in file, setting to None "
                dataset is None

            if dataset is None: # no dataset name is provided
                keys = f.keys()
                if len(keys) == 1: # there is only one dataset, use it
                    dataset = keys[0]
                else:
                    raise KeyError("No or wrong dataset name provided and the file has\
                    more than one")
            arr = f[dataset]
            parent.__init__(arr,**kwargs)

        def length(self):
            return self.data.shape[0]
        def frames(self):
            fn = self.pipeline()
            return itt.imap(fn, (f for f in self.data))


    class FSeq_hdf5_lsc(FrameSequence):
        "Class for hdf5 files written by pylsi software"
        def __init__(self, fname):
            parent = super(FSeq_hdf5_lsc, self)
            parent.__init__()
            f = h5py.File(fname, 'r')
            t = f['tstamps']
            self.tv = (t-t[0])/1e6 # relative time, in s
            self.dt = np.median(np.diff(self.tv))
            self.data = f['lsc']
            self.fns = []
        def length(self):
            return self.data.shape[0]
        def frames(self):
            fn = self.pipeline()
            return itt.imap(fn, (f for f in self.data))
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

    def fseq2h5(seq, name,compress_level=-1):
        # todo: add metadata, such as time and spatial scales
        if os.path.exists(name):
            sys.stderr.write("File exists, removing\n")
            os.remove(name)
        fid = h5py.File(name, 'w')
        L = seq.length()
        sh = seq.shape()
        chunkshape = tuple([1] + list(sh))
        fullshape = tuple([L] + list(sh))
        kwargs = dict()
        if compress_level > -1:
            kwargs['compression'] = 'gzip'
            kwargs['compression_opts'] = compress_level

        dset = fid.create_dataset('data', fullshape, dtype=seq[0].dtype,
                                  chunks = chunkshape, **kwargs)
        for k,f in enumerate(seq.frames()):
            dset[k,...] = f
            sys.stderr.write('\r writing frame %02d out of %03d'%(k,L))
        fid.close()
        return name

except ImportError as e:
    print "Import Error", e
        
        
        
            
