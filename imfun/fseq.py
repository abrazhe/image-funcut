### -------------------------------- ###
### Classes for sequences of frames ###
### -------------------------------- ###

from __future__ import division # a/b will always return float

import sys
import os
import re
import glob
import itertools as itt

import warnings

import numpy as np
#import tempfile as tmpf


_dtype_ = np.float64

import matplotlib
#matplotlib.use('Agg')


from matplotlib.pyplot import imread

#import quantities as pq

from imfun import lib
ifnot = lib.ifnot

class FrameSequence(object):
    "Base class for sequence of frames"
    def set_default_meta(self,ndim=None):
        self.meta = dict()
        scales = lib.alist_to_scale([(1,'')])
        self.meta['axes'] = scales

    def _get_scale(self):
	"""
        DO NOT USE
	Returns dx,dy and the flag whether the scale information has been set.
	If scale info hasn't been set, dx=dy=1
	"""
        warnings.warn("`_get_scale` is deprecated and will be removed from future releases")
        if hasattr(self, '_scale_set'):
            scale_flag = self._scale_set
            dx,dy, scale_flag = self.dx, self.dy, self._scale_set
        else:
            dx,dy,scale_flag = 1,1,None
        return dx, dy, scale_flag

    def _set_scale(self, dx=None, dy=None):
	"""DO NOT USE sets self.dx, self.dy scale information"""
        warnings.warn("`_get_scale` is deprecated and will be removed from future releases")        
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

    def std(self, axis=None):
	"""get standard deviation of the data"""
        a = self.as3darray()
        return float(a.std(axis))

    def data_range(self):
        """Return global range (`min`, `max`) values for the sequence"""
        # need this strange syntax for min/max for multi-channel images
        #rfn = lambda fn: lambda x: fn(fn(x, axis=0),axis=0)
        def rfn(fn): return lambda x: fn(fn(x, axis=0),axis=0)
        ranges = np.array([(rfn(np.min)(f), rfn(np.max)(f)) for f in self])
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

    def frame_idx(self,):
        """Return a vector of time stamps, if frame sequence is timelapse and
        dt is set, or just `arange(nframes)`"""
        L = len(self)
        scale, unit = self.meta['axes'][0]
        if unit == '': scale=1
        return np.arange(0, (L+2)*scale, scale)[:L]

    def mask_reduce(self, mask, fn=np.mean):
        """Return `1D` vector from a mask (or slice), by applying a reducing function
        R^n->R (average by default) within the mask in each frame.
        Function fn should be able to recieve `axis` optional argument"""
        return np.asarray([fn(f[mask],axis=0) for f in self])

    def softmask_reduce(self,mask, fn=np.mean):
	"""Same as mask_reduce, but pixel values are weighted by the mask values between 0 and 1"""
        sh = self.shape()
        mask2d = mask
        if len(sh) >2 :
            mask = np.dstack([mask]*sh[-1])
        return np.asarray([fn((f*mask)[mask2d>0],axis=0) for f in self])


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

	This is a more general (and sometimes faster) function than
	`self.mean_frame` or `self.max_project`. However, it requires more
	memory as it makes a call to self.as3darray, while the other two don't
	
	Parameters:
	  - `fslice`: (`int`, `tuple-like` or `None`) --
          [start,] stop [, step] go through these frames
	  - `fn` (`func`) -- function to apply (`np.mean` by default)

        Returns:
	  - `2D` `array` -- a projected frame
	"""
        sh = self.shape(crop)
	out = np.zeros(sh)
        #if len(sh)>2:
        #    fn = lambda a: np.mean(a, axis=0)
	for v,r,c in self.pix_iter(fslice=fslice,crop=crop):
	    out[r,c] = fn(v)
	return out

    def mean_frame(self, fslice=None):
        """Return average image over a number of frames (all by default).

	frame range is given as argument fslice. if it's int, use N first
	frames, if it's tuple-like, it can be of the form [start,] stop [,step]
	"""
        if fslice is None or isinstance(fslice, int):
            fslice = (fslice, )
        frameit = itt.imap(_dtype_, itt.islice(self.frames(), *fslice))
        res = np.copy(frameit.next())
        count = 0.0
        for k,frame in enumerate(frameit):
            res += frame
	    count += 1
        return res/(count)
    
    def max_project(self, fslice=None):
	"""Return max-projection image over a number of frames
        (all by default).

        see fseq.mean_frame docstring for details
	"""
        if fslice is None or isinstance(fslice,  int):
            fslice = (fslice, )
        frameit = itt.imap(_dtype_, itt.islice(self.frames(), *fslice))
        out = frameit.next() # fix it, it fails here
        for k,frame in enumerate(frameit):
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
        if isinstance(fslice, int):
            fslice = (fslice, )
        return itt.islice(fiter, *fslice)

    def as3darray(self,  fslice=None, crop=None,
                  dtype = _dtype_):
	"""Return the frames as a `3D` array.

	//An alternative way is to use the __getitem__ interface:
	//``data = np.asarray(fs[10:100])``
	

	Parameters:
	  - `fslice`: (`int`, `tuple-like` or `None`) --
          [start,] stop [, step] to go through frames
	  - `crop`: (`slice` or `None`) -- a crop (tuple of slices) to take from each frame
	  - `dtype`: (`type`) -- data type to use. Default, ``np.float64``

	Returns:
	  `3D` array `d`, where frames are stored in higher dimensions, such
	  that ``d[0]`` is the first frame, etc.
	"""
        if fslice is None or isinstance(fslice, int):
            fslice = (fslice, )
        shape = self.shape(crop)
	newshape = (len(self),) + shape
        out = lib.memsafe_arr(newshape, dtype)
        for k,frame in enumerate(itt.islice(self, *fslice)):
            out[k,...] = frame[crop]
        out = out[:k+1]
        if hasattr (out, 'flush'):
            out.flush()
        return out

    
    
    def loc_iter(self, pmask=None, fslice=None, rand=False,  crop=None):
        """Return iterator over pixel locations, which are True in `pmask` 

	Parameters:
	  - `pmask`: (2D `Bool` array or `None`) -- skip pixels where `mask` is
            `False` if `mask` is `None`, take all pixels
          - `fslice`: (`int`, `slice` or `None`) --
          [start,] stop [, step]  to go through frames
	  - `rand`: (`Bool`) -- whether to go through pixels in a random order
	  - `**kwargs`: keyword arguments to be passed to `self.as3darray`

	Yields:
	 tuples of `(v,row,col)`, where `v` is the time-series in a pixel at `row,col`
	
	"""
	sh = self.shape(crop)
        pmask = ifnot(pmask, np.ones(sh[:2], np.bool))
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
                yield row, col

    def pix_iter(self, pmask=None, fslice=None, rand=False, crop=None,  dtype=_dtype_):
        """Return iterator over time signals from each pixel.
        
	Parameters:
	  - `pmask`: (2D `Bool` array or `None`) -- skip pixels where `mask` is
            `False` if `mask` is `None`, take all pixels
          - `fslice`: (`int`, `slice` or `None`) -- [start,] stop [, step]  to go through frames
	  - `rand`: (`Bool`) -- whether to go through pixels in a random order
	  - `**kwargs`: keyword arguments to be passed to `self.as3darray`

	Yields:
	 tuples of `(v,row,col)`, where `v` is the time-series in a pixel at `row,col`
	"""
        arr = self.as3darray(fslice, crop=crop,dtype=dtype)
        for row, col in self.loc_iter(pmask=pmask,fslice=fslice,rand=rand,crop=crop):
            yield np.asarray(arr[:,row,col],dtype=dtype), row, col
        if hasattr(arr, 'flush'):
            arr.flush()
        del arr
	return
    
    def __len__(self):
        """Return number of frames in the sequence"""
        if not hasattr(self,'_length'):
            k = 0
            for _ in self:
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
	if 'dtype' in kwargs:
	    dtype = kwargs['dtype']
	else:
	    dtype = _dtype_
	dt = self.meta['axes'][0]['scale'] # todo: always convert to seconds
	arr = self.as3darray(**kwargs)
	sigma = tau/dt
	smooth =  ndimage.gaussian_filter1d(arr, sigma, axis=0)
	zi = np.where(np.abs(smooth) < 1e-6)
	out  = arr/smooth - 1.0
	out[zi] = 0
        newmeta = self.meta.copy()
	return FSeq_arr(out, meta=newmeta)
	

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
        if 'dtype' in kwargs:
	    dtype = kwargs['dtype']
	else:
	    dtype = _dtype_
	L = len(pwfn(np.random.randn(len(self))))
	#testv = pwfn(self.pix_iter(rand=True,**kwargs).next()[0])
	#L = len(testv)
	out = lib.memsafe_arr((L,) + self.shape(), dtype)
        for v, row, col in self.pix_iter(**kwargs):
	    if verbose:
		sys.stderr.write('\rworking on pixel (%03d,%03d)'%(row, col))
            out[:,row,col] = pwfn(v)
	    if hasattr(out, 'flush'):
		out.flush()

        ## assuming the dz is not changed. if it *is*, provide new meta
        ## in kwargs
        if 'meta' in kwargs:
            newmeta = kwargs['meta'].copy()
        else:
            newmeta = self.meta.copy() 
        return FSeq_arr(out, meta=newmeta)

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
            stop = len(self)
	if hasattr(self, 'data'):
	    vmin = ifnot(vmin, self.data_percentile(1)) # for scale
	    vmax = ifnot(vmax, self.data_percentile(99)) # for scale
	else:
	    vmin = ifnot(vmin, np.min(map(np.min, self.frames())))
	    vmax = ifnot(vmax, np.min(map(np.max, self.frames())))
        kwargs.update({'vmin':vmin, 'vmax':vmax})
	print path+base
        L = min(stop-start, len(self))
	fnames = []
        for i,frame in enumerate(self.frames()):
            if i < start: continue
            if i > stop: break
            ax.cla()
            ax.imshow(frame, aspect='equal', **kwargs)
            fname =  path + base + '%06d.png'%i
	    fnames.append(fname)
            if show_title:
                zscale = tuple(self.meta['axes'][0])
                ax.set_title('frame %06d (%3.3f %s)'%zscale)
            fig.savefig(fname)
            sys.stderr.write('\r saving frame %06d of %06d'%(i+1, L))
        plt.close()
	return fnames
    
    def export_movie_anim(self, video_name, fps=None, start=0, stop=None,
                          show_title=True, fig_size=(4,4),
                          bitrate=-1,
                          writer = 'avconv',
                          codec = None,
                          frame_on = False,
                          marker_idx = None,
                          vmin=None, vmax=None,**kwargs):
        """
        Create an mpg  movie from the frame sequence using mencoder.
        and mpl.Animation

	Parameters:
	  - `video_name`: (`str`) -- a name (without extension) for the movie to
	    be created
	  - `fps`: (`number`) -- frames per second.
             If None, use 10/self.meta['axes'][0][0]
          - `marker_idx`: (`array_like`) -- indices when to show a marker
            (e.g. for stimulation)
	  - `**kwargs` : keyword arguments to be passed to `self.export_png`
	"""
        warnings.warn("export_movie_anim is deprecated and will be removed in the future releases")
        import matplotlib as mpl
        mpl.use('Agg')
        from matplotlib import animation
        import matplotlib.pyplot as plt
        dz,zunits = tuple(self.meta['axes'][0])
        if fps is None:
            fps = max(1., 10./dz)
        if marker_idx is None:
            marker_idx = []

        if stop is None or stop == -1:
            stop = len(self)

	if hasattr(self, 'data'):
	    vmin = ifnot(vmin, self.data_percentile(1)) # for scale
	    vmax = ifnot(vmax, self.data_percentile(99)) # for scale
	else:
	    vmin = ifnot(vmin, np.min(map(np.min, self)))
	    vmax = ifnot(vmax, np.min(map(np.max, self)))

        kwargs.update({'vmin':vmin, 'vmax':vmax})
        L = min(stop-start, len(self))

        fig,ax = plt.subplots(1,1,figsize=fig_size)

        if np.ndim(self[start]) > 2:
            vmin = np.min(vmin)
            vmax = np.max(vmax)
            #lutfn = lambda f: np.clip(f, vmin,vmax)/vmax
            def lutfn(f): return np.clip((f-vmin)/(vmax-vmin), 0, 1)
        else:
            def lutfn(f): return f


        plh = ax.imshow(lutfn(self[start]), 
                        aspect='equal', **kwargs)
        if not frame_on:
            plt.setp(ax, frame_on=False, xticks=[],yticks=[])
        mytitle = ax.set_title('')
        marker = plt.Rectangle((2,2), 10,10, fc='red',ec='none',visible=False)
        ax.add_patch(marker)
        
        def _animate(framecount):
            tstr = ''
            k = framecount+start
            plh.set_data(lutfn(self[k]))
            if show_title:
                if zunits in ['sec','msec','s','usec', 'us','ms','seconds']:
                    tstr = ', time: %0.3f %s' %(k*dz,zunits)
                mytitle.set_text('frame: %04d'%k +tstr)
            if k in marker_idx:
                plt.setp(marker, visible=True)
            else:
                plt.setp(marker, visible=False)
            return plh,
        
        #anim = animation.FuncAnimation(fig, _animate, init_func=_init, frames=L, blit=True)
        anim = animation.FuncAnimation(fig, _animate, frames=L, blit=True)
        mencoder_extra_args=['-ovc', 'lavc', '-lavcopts', 'vcodec=mpeg4']
        if writer in animation.writers.list():
            # Set up formatting for the movie files
            Writer = animation.writers.avail[writer]
            w = Writer(fps=fps,bitrate=bitrate)#, metadata=dict(artist='Me'), bitrate=1800)
            anim.save(video_name, writer=w)
        else:
            anim.save(video_name+'.avi', writer='mencoder',
                      fps=fps, extra_args=mencoder_extra_args)
        
        #plt.close(anim._fig)
        plt.close(fig)
        #del anim, plh, ax # 
        return 


_x1=[(1,2,3), ('sec','um','um')] # two lists
_x2=[(1,'sec'), (2,'um'), (3,'um')] # plist
_x3=[1, 'sec', 2, 'um', 3, 'um'] # alist

def _empty_axes_meta(size=3):
    names = ("scale", "units")
    formats = ('float', "S10")
    x =  np.ones(size, dtype=dict(names=names,formats=formats))
    x['unit'] = ['']*size
    return x




class FSeq_arr(FrameSequence):
    """A FrameSequence class as a wrapper around a `3D` Numpy array
    """
    def __init__(self, arr, fns = None, meta=None):
        self.data = arr
        self.fns = ifnot(fns, [])
        if meta is None:
            self.set_default_meta(self)
        else:
            self.meta = meta.copy()
    def __len__(self):
        return len(self.data)
    def __getitem__(self, val):
        fn = self.pipeline()
        if isinstance(val,slice) or np.ndim(val) > 0:
            x = self.data[val]
            if not self.fns:
                return x
            out = np.zeros(x.shape,x.dtype)
            for j,f in enumerate(x):
                out[j] = fn(f)
            return out
        else:
            if val > len(self):
                raise IndexError("Requested frame number out of bounds")
            return fn(self.data[int(val)])

    def data_percentile(self, p):
        sh = self.shape()
        arr = self.data
        if len(sh) == 2:
            return  np.percentile(arr,p)
        else:
            return [np.percentile(arr[...,k],p) for k in range(sh[2])]

    def pix_iter(self, pmask=None, fslice=None, rand=False, crop=None,dtype=_dtype_):
	"Iterator over time signals from each pixel (FSeq_arr)"
        if not self.fns:
            for row, col in self.loc_iter(pmask=pmask,fslice=fslice,rand=rand):
	        v = self.data[:,row,col].copy()
		yield np.asarray(v, dtype=dtype), row, col
	else:
	    base = super(FSeq_arr,self)
	    for a in base.pix_iter(pmask=pmask,fslice=fslice, rand=rand,dtype=dtype):
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
        return itt.imap(fn, self.data)


def identity(x):
    return x

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
      - pattern: (`str`) -- glob-style pattern for file names. 
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


class FSeq_glob(FrameSequence):
    """A FrameSequence class as a wrapper around a set of files matching a
    glob-like pattern"""
    def __init__(self, pattern, ch=0, fns=None,
                 meta = None):
        self.pattern = pattern
        self.ch = ch
        self.fns = ifnot(fns, [])
	self.file_names = sorted_file_names(pattern)
        if meta is None:
            self.set_default_meta()
        else:
            self.meta = meta.copy()
    def __len__(self):
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
        if isinstance(val, slice)  or np.ndim(val) > 0:
            seq =  map(self.loadfn, self.file_names[val])
	    if self.ch is not None:
		seq = (img_getter(f, self.ch) for f in seq)
            return map(fn, seq)
        else:
            if val > len(self):
                raise IndexError("Requested frame number out of bounds")
	    frame = self.loadfn(self.file_names[val])
	    if self.ch is not None:
		frame = img_getter(frame, self.ch)
	    return fn(frame)
            
class FSeq_img(FSeq_glob):
    """FrameSequence around a set of image files"""
    def loadfn(self,y): return imread(y)

class FSeq_txt(FSeq_glob):
    """FrameSequence around a set of text-image files"""
    def loadfn(self,y): np.loadtxt(y)

## TODO: but npy can be just one array
class FSeq_npy(FSeq_glob):
    """FrameSequence around a set of npy files"""
    def loadfn(self, y): return np.load(y)

class FSeq_imgleic(FSeq_img):
    """FrameSequence around the image files created by LeicaSoftware.
    It is just a wrapper around FSeq_img, only it also looks for an xml
    file in Leica's format with the Job description
    """
    def __init__(self, pattern, ch=0, fns=None, xmlname = None,
                 meta=None):
        FSeq_glob.__init__(self, pattern, ch=ch, meta=meta)
        if xmlname is None:
            xmlname = pattern
        self.fns = ifnot(fns, [])
        try:
            from imfun import leica
            self.lp = leica.LeicaProps(xmlname)
            self.meta['axes'][1:3] = [(self.lp.dx,'um'), (self.lp.dy,'um')]
            if hasattr(self.lp, 'dt'):
                zscale = (self.lp.dt, 's')
            elif hasattr(self.lp, 'dz'):
                zscale = (self.lp.dz, 'um')
            else:
                zscale = (1, '')
            self.meta['axes'][0] = zscale

        except Exception as e:
            print "Got exception, ", e
            pass


#from imfun.MLFImage import MLF_Image
import MLFImage

class FSeq_mlf(FrameSequence):
    "FrameSequence class for MLF multi-frame images"
    def __init__(self, fname, fns = None):
        self.mlfimg = MLFImage.MLF_Image(fname)
        self.set_default_meta()
        dt = self.mlfimg.dt/1000.0
        self.meta['axes'][0] = (dt, 'sec')
        self.fns = ifnot(fns, [])
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
	#L = self.length()
	fn = self.pipeline()
        if isinstance(val, slice) or np.ndim(val) > 0:
	    indices = np.arange(self.mlfimg.nframes)
	    return itt.imap(fn, itt.imap(self.mlfimg.read_frame, indices[val]))
        else:
            if val > len(self):
                raise IndexError("Requested frame number out of bounds")
            return fn(self.mlfimg[val])
    def __len__(self):
        return self.mlfimg.nframes
    def pix_iter(self, pmask=None, fslice=None, rand=False, crop=None,dtype=_dtype_):
        "Iterator over time signals from each pixel, where pmask[pixel] is True"
	if not len(self.fns):
	    for row,col in self.loc_iter(pmask=pmask,fslice=fslice,rand=rand):
		v = self.mlfimg.read_timeslice((row,col))
		yield np.asarray(v, dtype=dtype), row, col
	else:
	    base = super(FSeq_mlf,self)
	    for a in base.pix_iter(pmask=pmask,fslice=fslice, rand=rand, dtype=dtype):
		yield a
		
import matplotlib.image as mpl_img
try:
    import PIL.Image as Image
    class FSeq_multiff(FrameSequence):
        "Class for multi-frame tiff files"
        def __init__(self, fname, fns = None, meta=None):
            self.set_default_meta()
            self.fns = ifnot(fns, [])
            self.im = Image.open(fname)
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
except ImportError:
    print "Cant define FSeq_multiff because can't import PIL"
                    
class FSeq_tiff_2(FSeq_arr):
    "Class for (multi-frame) tiff files, using tiffile.py by Christoph Gohlke"
    def __init__(self, fname, ch=None, flipv = False, fliph = False, **kwargs):
	import tiffile
	x = tiffile.imread(fname)
	parent = super(FSeq_tiff_2, self)
	parent.__init__(x, **kwargs)
        if isinstance(ch, basestring) and ch != 'all':
            ch = np.where([ch in s for s in 'rgb'])[0][()]
        if ch is not None and ch != 'all' and self.data.ndim > 3:
            self.data = np.squeeze(self.data[:,:,:,ch])
        if flipv:
            self.data = self.data[:,::-1,...]
        if fliph:
            self.data = self.data[:,:,::-1,...]



## -- MES files --
import mes

class FSeq_mes(FSeq_arr):
    def __init__(self, fname, record=None, ch=None, fns=None, verbose=False,
                 autocrop = True):
        """
        The following format is assumed:
        the mes file contains descriptions in fields like "Df0001",
        and actual images in fields like 'If0001_001', 'If0001_002'.
        These fields contain data for the red and green channel, accordingly
        The timelapse images are stored as NXM arrays, where N is one side of an image,
        and then columns iterate over the other image dimension and time.
        """
        self.fns = ifnot(fns, [])
        self._verbose=verbose

        if record is None:
            vars = filter(lambda v: v.is_supported, mes.load_file_info(fname))
            if len(vars):
                record = vars[0].record
            else:
                print "FSeq_mes: Can't find loadable records"
        elif isinstance(record,int):
            record = 'Df%04d'%record
        elif isinstance(record,str):
            if not ('Df' in record):
                record = 'Df%04d'%int(record)
        else:
            print "FSeq_mes: Unknown record definition format"

        self.mesrec = mes.load_record(fname, record, ch)
        self.data, self.meta = self.mesrec.load_data()
        if autocrop :
            nrows = self.data.shape[1]
            self.data = self.data[:,:,:nrows,...]

        


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
    if isinstance(path, np.ndarray):
	return FSeq_arr(path, *args, **kwargs)
    if isinstance(path, FrameSequence):
        return path
    ending = re.findall('[^*\.]+', path)[-1].lower()
    if ending == 'txt':
        handler = FSeq_txt
    elif ending == 'mes':
        handler = FSeq_mes
    elif ending == 'mlf':
        handler = FSeq_mlf
    elif ending == 'npy':
        handler =  FSeq_npy
    elif ending == 'h5':
        handler = FSeq_hdf5
    elif ending in images:  # A collection of images or a big tiff
        if '*' in path: # many files
            from imfun import leica
            xml_try = leica.get_xmljob(path)
            print '*******', path,     xml_try
            if 'xmlname' in kwargs or xml_try:
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
            print "The file %s has the following data sets:"%fname, f.keys()

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

    class FSeq_hdf5_lsc(FSeq_arr):
        "Class for hdf5 files written by pylsi software"
        def __init__(self, fname, fns = None):
            parent = super(FSeq_hdf5_lsc, self)
            #parent.__init__()
            f = h5py.File(fname, 'r')
            t = f['tstamps']
            self.tv = (t-t[0])/1e6 # relative time, in s
            dt = np.median(np.diff(self.tv))
            meta = {'axes': lib.alist_to_scale([dt,'sec'])}
            self.h5file = f # just in case we need it later
            parent.__init__(f['lsc'], fns=fns,meta=meta)

    def fseq2h5(seqlist, name,compress_level=-1,verbose=False):
        # todo: add metadata, such as time and spatial scales
        if os.path.exists(name):
            if verbose:
                sys.stderr.write("File exists, removing\n")
            os.remove(name)
        fid = h5py.File(name, 'w')
        L = np.sum(map(len, seqlist))
        sh = np.min([s.shape() for s in seqlist], axis=0)
        print 'shape:', sh
        chunkshape = tuple([1] + list(sh))
        fullshape = tuple([L] + list(sh))
        kwargs = dict()
        if compress_level > -1:
            kwargs['compression'] = 'gzip'
            kwargs['compression_opts'] = compress_level

        dset = fid.create_dataset('data', fullshape, dtype=seqlist[0][0].dtype,
                                  chunks = chunkshape, **kwargs)
        k = 0
        for seq in seqlist:
            for f in seq.frames():
                dset[k,...] = f
                if verbose:
                    sys.stderr.write('\r writing frame %02d out of %03d'%(k,L))
                k += 1    
        fid.close()
        return name


except ImportError as e: # couln't import h5py
    print "Import Error", e
        
        
        
            
## -- Load video from mp4 -- 
## requires cv2

try:
    import cv2
    def load_mp4(name, framerate=25., start_frame = None, end_frame=None, 
                 frame_fn = None,
                 **kwargs):
        vidcap = cv2.VideoCapture(name)
        out = []
        count = 0
        if frame_fn is None:
            frame_fn = lambda f:f
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = 1e9
        while True:
            success, image = vidcap.read()
            if (not success) or (count > end_frame) :
                break
            count +=1
            if count < start_frame:
                continue
            out.append(frame_fn(image))
        fs = open_seq(np.array(out), **kwargs)
        fs.meta['axes'][0] = (1./framerate, 's')
        return fs

    def mp4_to_hdf5(name, framerate=25., frame_fn=None, **kwargs):
        import sys
        if frame_fn is None:
            frame_fn = lambda f:f
        vidcap = cv2.VideoCapture(name)
        fid = h5py.File(name+'.h5', 'w')
        success, image = vidcap.read()
        image = frame_fn(image)
        if not success: return
        sh = image.shape
        chunkshape = tuple([1] + list(sh))
        data = fid.create_dataset('data', chunkshape, dtype=image.dtype,
                                  maxshape = (None,)+sh,
                                  chunks = chunkshape, **kwargs)
        #meta = fid.create_dataset('meta',
        frame_count = 0
        while True:
            sys.stderr.write('\r {}, {}, {}'.format(frame_count, data.shape, image.shape))
            data[frame_count,...] = image
            success, image = vidcap.read()
            image = frame_fn(image)
            if success:
                frame_count += 1                
                data.resize(frame_count+1,0)
            else:
                break
        fid.close()

except ImportError as e:
    print "Can't load OpenCV python bindings", e


def to_movie(fslist, video_name, fps=25, start=0,stop=None,
                    ncols = None,
                    figsize=None, figscale=4,
                    show_suptitle=True, titles = None,
                    writer='avconv', bitrate=2500,
                    frame_on=False, marker_idx=None,
                    clim = None, **kwargs):
    """
    Create an video file from the frame sequence using avconv or other writer.
    and mpl.Animation

    Parameters:
      - `video_name`: (`str`) -- a name (without extension) for the movie to
         be created
      - `fps`: (`number`) -- frames per second.
         If None, use 10/self.meta['axes'][0][0]
      - `marker_idx`: (`array_like`) -- indices when to show a marker
        (e.g. for stimulation)
      - `**kwargs` : keyword arguments to be passed to `self.export_png`
    """
    from matplotlib import animation
    import matplotlib.pyplot as plt
    import gc

    plt_interactive = plt.isinteractive()
    plt.ioff() # make animations in non-interactive mode
    if isinstance(fslist, FrameSequence):
        fslist = [fslist]
    
    marker_idx = ifnot(marker_idx, [])
    stop = ifnot(stop, np.min(map(len, fslist)))
    L = stop-start

    dz,zunits = tuple(fslist[0].meta['axes'][0]) # units of the first frame sequence are used

    lutfns = []
    for fs in fslist:
        if clim is not None:
            vmin, vmax = clim
        else:
            bounds = fs.data_percentile((1,99)) if hasattr(fs,'data') else fs.data_range()
            bounds = np.array(bounds).reshape(-1,2)
            vmin, vmax = np.min(bounds[:,0],0), np.max(bounds[:,1],0)
        #print vmin,vmax
        lutfn = lambda f: np.clip((f-vmin)/(vmax-vmin), 0, 1)
        lutfns.append(lutfn)


    #----------------------
    if ncols is None:
        nrows, ncols = lib.guess_gridshape(len(fslist))
    else:
        nrows = int(np.ceil(len(fslist)/ncols))
    sh = fslist[0].shape()
    aspect = float(sh[0])/sh[1]
    header_add = 0.5
    figsize = ifnot (figsize, (figscale*ncols/aspect, figscale*nrows + header_add)) 

    fig, axs = plt.subplots(nrows, ncols, figsize=figsize)

    titles = ifnot(titles, ['']*len(fslist))
    if len(titles) < len(fslist):
        titles = list(titles) + ['']*(len(fslist)-len(titles))

    if 'aspect' not in kwargs:
        kwargs['aspect']='equal'
    if 'cmap' not in kwargs:
        kwargs['cmap'] = 'gray'

    views = []
    for fs,lut,title,ax in zip(fslist, lutfns, titles, np.ravel(axs)):
        view = ax.imshow(lut(fs[start]), **kwargs)
        views.append(view)
        ax.set_title(title)
        if not frame_on:
            plt.setp(ax,frame_on=False,xticks=[],yticks=[])
        marker = plt.Rectangle((2,2), 10,10, fc='red',ec='none',visible=False)
        ax.add_patch(marker)
        
    
    mytitle = plt.suptitle('')
    plt.tight_layout()
    
    
    #----------------------

    def _animate(framecount):
        tstr = ''
        k = start + framecount
        for view, fs, lut in zip(views, fslist, lutfns):
            view.set_data(lut(fs[k]))
        if show_suptitle:
            if zunits == ['sec','msec','s','usec', 'us','ms','seconds']:
                tstr = ', time: %0.3f %s' %(k*dz,zunits) #TODO: use in py3 way
            mytitle.set_text('frame %04d'%k + tstr)
        if k in marker_idx:
            plt.setp(marker, visible=True)
        else:
            plt.setp(marker, visible=False)
        return views

    anim = animation.FuncAnimation(fig, _animate, frames=L, blit=True)
    Writer = animation.writers.avail[writer]
    w = Writer(fps=fps,bitrate=bitrate)
    anim.save(video_name, writer=w)

    fig.clf()
    plt.close(fig)
    plt.close('all')
    del anim, w, axs, _animate
    gc.collect()
    if plt_interactive:
        plt.ion()
    return
    

    
    
