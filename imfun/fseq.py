### -------------------------------- ###
### Classes for sequences of frames ###
### -------------------------------- ###

 # a/b will always return float

import sys
import os
import re
import glob
import itertools as itt
import inspect

import zipfile
import fnmatch


from skimage import io as skio
from skimage.external import tifffile





import warnings

import numpy as np
from scipy import ndimage
import collections
#import tempfile as tmpf

try:
    from numba import jit
except ImportError:
    def jit(fn):
        return fn


_dtype_ = np.float64

#import matplotlib
#matplotlib.use('Agg')


from matplotlib.pyplot import imread

#import quantities as pq


from imfun.external.physics import Q

from . import core
#from . import ui
from .core import ifnot
from .core import fnutils as fu
from .core import array_handling as ah
from .core import units
from .core.units import QS

class FrameStackMono(object):
    "Base class for a stack {stream, sequence} of single-channel frames"
    def set_default_meta(self,ndim=None):
        self.meta = dict()
        scales = units.alist_to_scale([(1,'_')])
        self.meta['axes'] = scales
        #self.meta['axes'] = [Q(1,'_') for i in range(3)]
        self.meta['channel'] = None

    @property
    def pipeline(self):
        """Return the composite function to process frames based on self.frame_filters"""
        return fu.flcompose(_identity, *self.frame_filters)

    def std(self, axis=None):
        """get standard deviation of the data"""
        a = self.as3darray()
        return float(a.std(axis))

    def data_range(self):
        """Return global range (`min`, `max`) values for the sequence"""
        def rfn(fn): return lambda x: fn(fn(x, axis=0),axis=0)
        ranges = np.array([(np.min(f), np.max(f)) for f in self])
        minv,maxv = np.min(ranges[:,0],axis=0), np.max(ranges[:,1],axis=0)
        return np.array([minv,maxv]).T # why do I transpose?

    def data_percentile(self, p):
        """Return a percentile `p` value on data.

        Parameters:
          - `p` : float in range of [0,100] (or sequence of floats)
             Percentile to compute which must be between 0 and 100 inclusive.

        """
        arr = self[:]
        return  np.percentile(arr,p)

    def frame_idx(self,):
        """Return a vector of time stamps, if frame sequence is timelapse and
        dt is set, or just `arange(nframes)`"""
        L = len(self)
        scale = self.meta['axes'][0].value
        #if unit == '': scale=1
        return np.arange(0, (L+2)*scale, scale)[:L]

    def mask_reduce(self, mask, fn=np.mean):
        """Return `1D` vector from a mask (or slice), by applying a reducing function
        R^n->R (average by default) within the mask in each frame.
        Function fn should be able to recieve `axis` optional argument"""
        crop = ndimage.find_objects(mask)[0]
        return np.asarray([fn(f[crop][mask[crop]],axis=0) for f in self])

    def multi_mask_reduce(self, masks,fn=np.mean):
        """
        Same as mask_reduce, but for multiple masks simultaneously
        """
        crops = [ndimage.find_objects(m)[0] for m in masks]
        return np.asarray([[fn(f[crop][mask[crop]],axis=0) for crop,mask in zip(crops,masks)] for f in self])


    def softmask_reduce(self,mask, fn=np.mean):
        """Same as mask_reduce, but pixel values are weighted by the mask values between 0 and 1"""
        return np.asarray([fn((f*mask)[mask>0],axis=0) for f in self])

    def multi_softmask_reduce(self, masks, fn=np.mean):
        '''
        Same as softmask_reduce, but for multiple masks simultaneously
        '''
        return np.asarray([[fn((f*mask)[mask>0],axis=0) for mask in masks] for f in self])


    def frame_slices(self, crop=None):
        """Return iterator over subframes (slices defined by `crop` parameter).
        When `crop` is `None`, full frames are returned
        """
        if crop is None:
            return self.frames()
        else:
            return (f[crop] for f in self)

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
        sh = self.get_frame_shape(crop)
        out = np.zeros(sh)
        if isinstance(fslice, collections.Callable):
            fn = fslice
            fslice = None
        #if len(sh)>2:
        #    fn = lambda a: np.mean(a, axis=0)
        for v,r,c in self.pix_iter(fslice=fslice):
            out[r,c] = fn(v)
        return out

    def mean_frame(self, fslice=None):
        """Return average image over a number of frames (all by default).

        frame range is given as argument fslice. if it's int, use N first
        frames, if it's tuple-like, it can be of the form [start,] stop [,step]
        """
        if fslice is None or isinstance(fslice, (np.integer, int)):
            if fslice is not None:
                fslice = int(fslice)
            fslice = (fslice, )
        frameit = map(_dtype_, itt.islice(self.frames(), *fslice))
        res = np.copy(next(frameit))
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
        frameit = map(_dtype_, itt.islice(self.frames(), *fslice))
        out = next(frameit) # fix it, it fails here
        for k,frame in enumerate(frameit):
            out = np.max([out, frame], axis=0)
        return out

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
        shape = self.get_frame_shape(crop)
        newshape = (len(self),) + shape
        out = ah.memsafe_arr(newshape, dtype)
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
        sh = self.get_frame_shape(crop)
        pmask = ifnot(pmask, np.ones(sh[:2], np.bool))
        nrows, ncols = sh[:2]
        rcpairs = [(r,c) for r in range(nrows) for c in range(ncols)]
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
            ## asarray to convert from memory-mapped array
            yield np.asarray(arr[:,row,col],dtype=dtype), row, col
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

    def get_frame_shape(self, crop=None):
        "Return the shape of frames in the sequence"
        return next(self.frame_slices(crop)).shape
    frame_shape = property(get_frame_shape)

    def _norm_mavg(self, tau=90., **kwargs):
        "Return normalized and temporally smoothed frame sequence"
        if 'dtype' in kwargs:
            dtype = kwargs['dtype']
        else:
            dtype = _dtype_
        dt = self.meta['axes'][0]
        dt = dt.to('s')
        arr = self.as3darray(**kwargs)
        sigma = tau/dt
        smooth =  ndimage.gaussian_filter1d(arr, sigma, axis=0)
        zi = np.where(np.abs(smooth) < 1e-6)
        out  = arr/smooth - 1.0
        out[zi] = 0
        newmeta = self.meta.copy()
        return FStackM_arr(out, meta=newmeta)


    def pw_transform(self, pwfn, verbose=False, **kwargs):
        """Spawn another frame sequence, pixelwise applying a user-provided
        function.

        Parameters:
          - `pwfn`: (`func`) -- a ``f(vector) -> vector`` function
          - `verbose`: (`Bool`) -- whether to be verbose while going through
            the pixels
          - `**kwargs``: keyword arguments to be passed to `self.pix_iter`
        """
        #nrows, ncols = self.frame_shape[:2]
        if 'dtype' in kwargs:
            dtype = kwargs['dtype']
        else:
            dtype = _dtype_
        L = len(pwfn(np.random.randn(len(self))))
        #testv = pwfn(self.pix_iter(rand=True,**kwargs).next()[0])
        #L = len(testv)
        out = ah.memsafe_arr((L,) + self.frame_shape, dtype)
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
        return FStackM_arr(out, meta=newmeta)

    def zoom(self, scales):
        """Zoom in or out so self.meta['axes'] match provided scales"""
        data = self[:]
        scales0 = self.meta['axes']['value']
        #TODO: rewrite this 
        #scales = [s.to(s0.unit) for s0,s in zip(scales0,scales)]
        zoom_factors = [s0/s for s0,s in zip(scales0,scales)]
        new_data = ndimage.zoom(data, zoom_factors)
        new_fs = FStackM_arr(new_data, meta=self.meta)
        new_scales = units.alist_to_scale([(1,'_')])
        new_scales['value'] =scales
        new_scales['unit'] = self.meta['axes']['unit']
        new_fs.meta['axes'] = new_scales
        return new_fs

    def to_hdf5(self, out_name,channel='', mode='a'):
        "Export sequence of frames to an HDF5 file with a dataset name as specified in argument 'channel'"
        import h5py
        f = h5py.File(out_name,mode)
        if channel == '':
            if 'channel' in self.meta and self.meta['channel'] is not None:
                channel = self.meta['channel']
        if channel == '':
            channel = 'frames'
                
        f.create_dataset(channel, data=self[:])
        f.close()
        pass


_x1=[(1,2,3), ('sec','um','um')] # two lists
_x2=[(1,'sec'), (2,'um'), (3,'um')] # alist
_x3=[1, 'sec', 2, 'um', 3, 'um'] # plist

def _empty_axes_meta(size=3):
    names = ("scale", "units")
    formats = ('float', "S10")
    x =  np.ones(size, dtype=dict(names=names,formats=formats))
    x['units'] = ['']*size
    return x

class FStackM_dummy(FrameStackMono):
    """
A FrameStackMono class where a single frame is repeated many times
    """
    def __init__(self, frame, L, frame_filters=None,meta=None):
        self.L = L
        self.frame = frame.copy()
        self.frame_filters = ifnot(frame_filters, [])
        self.ffs = self.frame_filters
        if meta is None:
            self.set_default_meta(self)
        else:
            self.meta = meta.copy()
    def __len__(self):
        return self.L
    def __getitem__(self,val):
        fn = self.pipeline
        if isinstance(val,slice)  or np.ndim(val) > 0:
            x = np.array([self.frame for i in np.r_[val]])
            dtype = fn(x[0]).dtype
            if not self.frame_filters:
                return x
            out = np.zeros(x.shape,dtype)
            for j,f in enumerate(x):
                out[j] = fn(f)
            return out
        else:
            if val >= len(self):
                raise IndexError("Requested frame number out of bounds")
            return fn(self.frame)
    def frames(self):
        fn = self.pipeline
        return map(fn, itt.repeat(self.frame,self.L))
        
        

class FStackM_arr(FrameStackMono):
    """A FrameStackMono class as a wrapper around a `3D` Numpy array
    """
    def __init__(self, arr, frame_filters = None, meta=None):
        self.data = arr
        self.frame_filters = ifnot(frame_filters, [])
        self.ffs = self.frame_filters
        if meta is None:
            self.set_default_meta(self)
        else:
            self.meta = meta.copy()
    def __len__(self):
        return len(self.data)
    def __getitem__(self, val):
        fn = self.pipeline
        if isinstance(val,slice) or np.ndim(val) > 0:
            x = self.data[val]
            dtype = fn(x[0]).dtype
            if not self.frame_filters:
                return x
            out = np.zeros(x.shape,dtype)
            for j,f in enumerate(x):
                out[j] = fn(f)
            return out
        else:
            if val >= len(self):
                raise IndexError("Requested frame number out of bounds")
            return fn(self.data[int(val)])

    def pix_iter(self, pmask=None, fslice=None, rand=False, crop=None,dtype=_dtype_):
        "Iterator over time signals from each pixel (FSM_arr)"
        if not self.frame_filters:
            for row, col in self.loc_iter(pmask=pmask,fslice=fslice,rand=rand):
                v = self.data[:,row,col].copy()
                yield np.asarray(v, dtype=dtype), row, col
        else:
            base = super(FStackM_arr,self)
            for a in base.pix_iter(pmask=pmask,fslice=fslice, rand=rand,dtype=dtype):
                yield a

    def frames(self):
        """
        Return iterator over frames.

        The composition of functions in `self.frame_filters` list is applied to each
        frame. By default, this list is empty.  Examples of function "hooks"
        to put into `self.frame_filters` are filters from ``scipy.ndimage``.
        """
        fn = self.pipeline
        return map(fn, self.data)


def _identity(x):
    return x

def sorted_file_names(pattern):
    "Return a sorted list of file names matching a pattern"
    x = glob.glob(pattern)
    return sorted(x)

def iter_files(pattern, loadfn):
    """Return iterator over data frames, file names matching a pattern,
    loaded by a user-provided function loadfn
    """
    return map(loadfn, sorted_file_names(pattern))


def img_getter(frame, ch):
    """A wrapper to extract color channel from image files.
    :returns: 2D matrix with intensity values
    """
    if len(frame.shape) > 2:
        return frame[:,:,ch]
    else:
        return frame


def __fseq_from_glob(pattern, ch=0, loadfn=np.load):
    """Return sequence of frames from filenames matching a glob.

    Parameters:
      - pattern: (`str`) -- glob-style pattern for file names.
      - ch: (`int` or `None`) -- color channel to extract if a number, all colors if `None`.
      - loadfn: (`func`) -- a function to load data from a file by its name [`np.load`].

    Returns:
      - iterator over frames. (`2D`)
    """
    return map(lambda frame: img_getter(frame, ch), iter_files(pattern, loadfn))

class FStackM_collection(FrameStackMono):
    """A FrameStackMono class as a wrapper around a set of files
    provided as a collection of file names or a glob-like pattern"""
    def __init__(self, names, ch=0, frame_filters=None,
                 meta = None):
        #self.pattern = pattern
        self.ch = ch
        self.frame_filters = ifnot(frame_filters, [])
        self.ffs = self.frame_filters
        if isinstance(names, str):
            self.file_names = sorted_file_names(names)
        else:
            self.file_names = names[:] # don't automatically sort
        if meta is None:
            self.set_default_meta()
        else:
            self.meta = meta.copy()
    def __len__(self):
        return len(self.file_names)

    def frames(self):
        """
        Return iterator over frames.

        The composition of functions in `self.frame_filters`
        list is applied to each frame. By default, this list is empty. Examples
        of function "hooks" to put into `self.frame_filters` are ``core.baselines.DFoSD``,
        ``core.baselines.DFoF`` or functions from ``scipy.ndimage``.
        """
        fn = self.pipeline
        seq = (img_getter(self.loadfn(name),self.ch) for name in self.file_names)
        return map(fn, seq)

    def __getitem__(self, val):
        fn = self.pipeline
        if isinstance(val, slice)  or np.ndim(val) > 0:
            seq = (img_getter(self.loadfn(name),self.ch) for name in self.file_names[val])
            return list(map(fn, seq))
        else:
            if val > len(self):
                raise IndexError("Requested frame number out of bounds")
            frame = self.loadfn(self.file_names[val])
            frame = img_getter(frame, self.ch)
            return fn(frame)

class FStackM_img(FStackM_collection):
    """FrameStackMono around a set of image files"""
    def loadfn(self,y): return skio.imread(y)

class FStackM_imgzip(FStackM_collection):
    def loadfn(self,name):
        return skio.imread(self.zf.open(name))

class FStackM_txt(FStackM_collection):
    """FrameStackMono around a set of text-image files"""
    def loadfn(self,y): return np.loadtxt(y)

class FStackM_ageom(FStackM_collection):
    def loadfn(self, name):
        fid =open(name,'rb')
        sh = np.fromfile(fid, np.int32, 2);
        v = np.fromfile(fid, np.float32, np.int(np.prod(sh)))
        fid.close()
        return v.reshape(sh)

## TODO: but npy can be just one array
class FStackM_npy_collection(FStackM_collection):
    """FrameSequence around a set of npy files"""
    def loadfn(self, y): return np.load(y)

class FStackM_npy(FStackM_arr):
    def __init__(self, name, frame_filters = None, meta=None):
        self.data = np.load(name)
        self.frame_filters = ifnot(frame_filters, [])
        self.ffs = self.frame_filters
        if meta is None:
            self.set_default_meta(self)
        else:
            self.meta = meta.copy()

class FStackM_imgleic(FStackM_img):
    """FrameSequence around the image files created by LeicaSoftware.
    It is just a wrapper around FStackM_img, only it also looks for an xml
    file in Leica's format with the Job description
    """
    def __init__(self, pattern, ch=0, frame_filters=None, xmlname = None,
                 meta=None):
        FStackM_collection.__init__(self, pattern, ch=ch, meta=meta)
        if xmlname is None:
            xmlname = pattern
        self.frame_filters = ifnot(frame_filters, [])
        self.ffs = self.frame_filters
        try:
            from imfun.io import leica
            self.lp = leica.LeicaProps(xmlname)
            #self.meta['axes'][1:3] = [QS(self.lp.dx,'um'), QS(self.lp.dy,'um')]
            self.meta['axes'][1:3] = units.alist_to_scale([(self.lp.dx,'um'), (self.lp.dy,'um')])
            if hasattr(self.lp, 'dt'):
                zscale = QS(self.lp.dt, 's')
            elif hasattr(self.lp, 'dz'):
                zscale = QS(self.lp.dz, 'um')
            else:
                zscale = QS(1, '_')
            self.meta['axes'][0] = zscale

        except Exception as e:
            print("Got exception, ", e)
            pass


#from imfun.MLFImage import MLF_Image
from imfun.io import ioraw

class FStackM_plsi(FrameStackMono):
    "FrameStackMono class for LaserSpeckle multi-frame images"
    def __init__(self, fname, frame_filters = None):
        self.plsimg = ioraw.PLSI(fname)
        self.set_default_meta()
        dt = self.plsimg.dt/1000.0
        self.meta['axes'][0] = QS(dt, 's')
        self.frame_filters = ifnot(frame_filters, [])
        self.ffs = self.frame_filters

    def frames(self,):
        """
        Return iterator over frames.

        The composition of functions in `self.frame_filters`
        list is applied to each frame. By default, this list is empty. Examples
        of function "hooks" to put into `self.frame_filters` are ``core.baselines.DFoSD``,
        ``core.baselines.DFoF`` or functions from ``scipy.ndimage``.
        """

        fn = fu.flcompose(_identity, *self.frame_filters)
        return map(fn,self.plsimg.frame_iter())

    def __getitem__(self, val):
        #L = self.length()
        fn = self.pipeline
        if isinstance(val, slice) or np.ndim(val) > 0:
            indices = np.arange(self.plsimg.nframes)
            if not self.frame_filters:
                return self.plsimg[val]
            else:
                return np.array(list(map(fn, map(self.plsimg.read_frame, indices[val]))))
                #return itt.imap(fn, itt.imap(self.plsimg.read_frame, indices[val]))
        else:
            if val > len(self):
                raise IndexError("Requested frame number out of bounds")
            return fn(self.plsimg[val])
    def __len__(self):
        return self.plsimg.nframes
    def pix_iter(self, pmask=None, fslice=None, rand=False, crop=None,dtype=_dtype_):
        "Iterator over time signals from each pixel, where pmask[pixel] is True"
        if not len(self.frame_filters):
            for row,col in self.loc_iter(pmask=pmask,fslice=fslice,rand=rand):
                v = self.plsimg.read_timeslice((row,col))
                yield np.asarray(v, dtype=dtype), row, col
        else:
            base = super(FStackM_plsi,self)
            for a in base.pix_iter(pmask=pmask,fslice=fslice, rand=rand, dtype=dtype):
                yield a


## --- TIFF files ---
## Beause there is skimage.io, and I have skimage as a dependency, there is no point
## in not using it


## import matplotlib.image as mpl_img
## try:
##     import PIL.Image as Image
##     class FSeq_multiff(FrameStackMono):
##         "Class for multi-frame tiff files"
##         def __init__(self, fname, frame_filters = None, meta=None):
##             self.set_default_meta()
##             self.frame_filters = ifnot(frame_filters, [])
##             self.im = Image.open(fname)
##         def frames(self, count=0):
##             """
##             Return iterator over frames.

##             The composition of functions in `self.frame_filters`
##             list is applied to each frame. By default, this list is empty. Examples
##             of function "hooks" to put into `self.frame_filters` are functions from ``scipy.ndimage``.
##             """
##             fn = self.pipeline
##             while True:
##                 try:
##                     self.im.seek(count)
##                     count += 1
##                     yield fn(mpl_img.pil_to_array(self.im))
##                 except EOFError:
##                     break
## except ImportError:
##     print "Cant define FSeq_multiff because can't import PIL"

## class FSeq_tiff_2(FStackM_arr):
##     "Class for (multi-frame) tiff files, using tiffile.py by Christoph Gohlke"
##     def __init__(self, fname, ch=None, flipv = False, fliph = False, **kwargs):
##      import tiffile
##      x = tiffile.imread(fname)
##      parent = super(FSeq_tiff_2, self)
##      parent.__init__(x, **kwargs)
##         if isinstance(ch, basestring) and ch != 'all':
##             ch = np.where([ch in s for s in 'rgb'])[0][()]
##         if ch not in (None, 'all') and self.data.ndim > 3:
##             self.data = np.squeeze(self.data[:,:,:,ch])
##         if flipv:
##             self.data = self.data[:,::-1,...]
##         if fliph:
##             self.data = self.data[:,:,::-1,...]





## -- MES files --
from imfun.io import mes

class FStackM_mes(FStackM_arr):
    def __init__(self, fname, record, ch='r', frame_filters=None, verbose=False,
                 autocrop = True):
        """
        The following format is assumed:
        the mes file contains descriptions in fields like "Df0001",
        and actual images in fields like 'If0001_001', 'If0001_002'.
        These fields contain data for the red and green channel, accordingly
        The timelapse images are stored as NxM arrays, where N is one side of an image,
        and then columns iterate over the other image dimension and time.
        """
        self.frame_filters = ifnot(frame_filters, [])
        self.ffs = self.frame_filters
        self._verbose=verbose


        self.mesrec = mes.load_record(fname, record)
        self.ch = ch
        self.data, self.meta = self.mesrec.load_data(ch)
        self.meta['channel'] = ch
        if autocrop :
            nrows = self.data.shape[1]
            self.data = self.data[:,:,:nrows,...]

## -- End of MES files --



class FStackColl(object):
    "Class for a collection (container) of single-channel frame stacks. E.g. Multichannel data"
    def __init__(self, stacks, meta=None):
        self.stacks = list(stacks)
        for k,s in enumerate(self.stacks):
            if not s.meta['channel']:
                s.meta['channel'] = str(k)

        # TODO: harmonize metadata
        if meta is None:
            self.meta = stacks[0].meta.copy()
        else:
            self.meta=meta
    @property
    def nCh(self):
        return len(self.stacks)
    def append(self, stream):
        if not stream.meta['channel']:
            stream_names = [s.meta['channel'] for s in self.stacks]
            for k in range(1000):
                if str(k) not in stream_names:
                    stream.meta['channel'] = str(k)
                    break
        self.stacks.append(stream)
    def pop(self, n=-1):
        return self.stacks.pop(n)
    def _propagate_call(self, call, *args, **kwargs):
        out = [getattr(s,call)(*args, **kwargs) for s in self.stacks]
        return np.stack(out,-1)
    def data_range(self, *args, **kwargs):
        return self._propagate_call('data_range', *args,**kwargs)
    def data_percentile(self, *args, **kwargs):
        return self._propagate_call('data_percentile', *args,**kwargs)
    @property
    def frame_shape(self):
        return self[0].shape
    def zoom(self,scales):
        zoomed_stacks = [s.zoom(scales) for s in self.stacks]
        out =  FStackColl(zoomed_stacks)
        out.meta = zoomed_stacks[0].meta
        return out
    def mean_frame(self,*args,**kwargs):
        mfs = [s.mean_frame(*args,**kwargs) for s in self.stacks]
        return np.stack(mfs, -1)
    def max_project(self,*args,**kwargs):
        return self._propagate_call('max_project',*args,**kwargs)
    def time_project(self,*args,**kwargs):
        projs = [s.time_project(*args,**kwargs) for s in self.stacks]
        return np.stack(projs,-1)
    def __len__(self):
        return np.min(list(map(len, self.stacks)))
    def __getitem__(self,val):
        x =  np.array([s[val] for s in self.stacks])
        return np.stack(x,-1)
    def order_stacks(self, name_order):
        ordered_stacks = []
        for name in name_order:
            match = [s for s in self.stacks if s.meta['channel'].lower() in name]
            if match :
                ordered_stacks.append(match[0])
        other_stacks = [s for s in self.stacks if s not in ordered_stacks]
        self.stacks = ordered_stacks + other_stacks
    def to_hdf5(self, out_name):
        import h5py
        f = h5py.File(out_name,'w')
        for i,stack in enumerate(self.stacks):
            if not 'channel' in stack.meta or stack.meta['channel'] is None:
                stack.meta['channel'] = ''
            if stack.meta['channel']=='':
                stack.meta['channel']='ch%d'%i
                
            f.create_dataset(stack.meta['channel'], data=stack[:])
        f.close()
        pass


    

def from_images(path,flavor=None,**kwargs):
    """Load a Multichannel FStack collection from a collection of images.
    When `flavor` is set to 'leica', try to load Leica XML metadata
    """
    meta = 'meta' in kwargs and kwargs['meta'] or None
    cmap = dict(r=0,g=1,b=2)
    if not 'ch' in kwargs or kwargs['ch'] is None:
        stacks = [construct_with_kwargs(FStackM_img, path, ch=channel, **kwargs) for channel in (0,1,2)]
        obj =  FStackColl(stacks,meta=meta)
    else:
        ch = kwargs.pop('ch')
        if isinstance(ch, str): # convert from rgb string to 012
            ch = cmap[ch]
        obj = construct_with_kwargs(FStackM_img, path, ch=ch,**kwargs)
    if isinstance(flavor, str) :
        if flavor.lower() == 'leica':
            attach_leica_metadata(obj, path)
    return obj


from imfun.io import leica
def attach_leica_metadata(obj, path, xmlname=None):
    if not xmlname:
        xmlname = leica.get_xmljob(path)
    if xmlname:
        lp = leica.LeicaProps(xmlname)
        obj.lp = lp
        obj.meta['axes'][1:3] = units.alist_to_scale([(lp.dx,'um'), (lp.dy,'um')])
        if hasattr(lp, 'dt'):
            zscale = QS(lp.dt, 's')
        elif hasattr(lp, 'dz'):
            zscale = QS(lp.dz, 'um')
        else:
            zscale = QS(1, '_')
        obj.meta['axes'][0] = zscale
        colors = 'rgb'
        if isinstance(obj, FStackColl):
            for k,stream in enumerate(obj.stacks):
                stream.meta = obj.meta.copy()
                stream.meta['channel'] = colors[k]


# def from_images_leica(path, *args, **kwargs):
#     from imfun import leica
#     xml_try = leica.get_xmljob(path)
#     if 'xmlname' in kwargs or xml_try:
#         handler =  FStackM_imgleic
#     else:
#         handler =  FStackM_img
#     stacks = [handler(path, ch=channel,*args,**kwargs) for channel in (0,1,2)]
#     return FStackColl(stacks)


def from_mes(path, record=None, ch=None, **kwargs):
    channel_order = ['pmtur', 'pmtug', 'pmtub']
    if record is None:
        vars = [v for v in mes.load_file_info(path) if v.is_supported]
        if len(vars):
            record = vars[0].record
        else:
            print("FSeq_mes: Can't find loadable records")
    elif isinstance(record,int):
        record = 'Df%04d'%record
    elif isinstance(record,str):
        if not ('Df' in record):
            record = 'Df%04d'%int(record)
    else:
        print("FSeq_mes: Unknown record definition format")

    mesrec = mes.load_record(path, record)
    channels = np.unique(mesrec.channels)
    if ch is None:
        stacks = [construct_with_kwargs(FStackM_mes, path, record=record, ch=ch, **kwargs) for ch in channels]
        obj =  FStackColl(stacks)
        obj.order_stacks(channel_order)
        return obj
    else:
        return construct_with_kwargs(FStackM_mes, path, record, ch=ch,**kwargs)


def from_plsi(path, *args, **kwargs):
    """Load LaserSpeckle frame stack.
    For arguments, see FStackCollackM_plsi constructor
    """
    return construct_with_kwargs(FStackM_plsi, path, *args, **kwargs)

def from_h5(path, *args, **kwargs):
    "Load frame stack from a generic HDF5 file"
    return construct_with_kwargs(FStackM_hdf5, path, *args, **kwargs)



def from_array(data,ch=None,  *args,**kwargs):
    
    sh = data.shape
    cmap = dict(r=0,g=1,b=2)
    
    if np.ndim(data)>3: #(Multichannel)
        #assume smallest dimension is the number of channels
        k = np.argmin(sh)
        nch = sh[k]
        channels = list(map(np.squeeze, np.split(data, nch, k)))
        print(len(channels), channels[0].shape)
        if ch is None:
            stacks = [construct_with_kwargs(FStackM_arr, c, *args, **kwargs) for c in channels]
            meta = 'meta' in kwargs and kwargs['meta'] or None
            return FStackColl(stacks,meta=meta)
        else:
            if isinstance(ch, str): # convert from rgb string to 012
                ch = cmap[ch]
            print('Channel:', ch)
            return construct_with_kwargs(FStackM_arr, channels[ch], *args, **kwargs)
    else:
        return construct_with_kwargs(FStackM_arr,data, *args,**kwargs)

def from_npy(path, *args, **kwargs):
    data = np.load(path)
    return from_array(data, *args, **kwargs)

def from_tiff(path, flavor=None, **kwargs):
    data = tifffile.imread(path)
    obj = from_array(data, **kwargs)
    if isinstance(flavor, str) and  flavor.lower() == 'olympus':
        attach_olympus_metadata(obj, path)
    return obj

def attach_olympus_metadata(obj, path):
    from imfun.io import olympus
    dye_order = ['dsred','ogb','egfp','trict']
    path_base, ext = os.path.splitext(path)
    meta = olympus.parse_meta_general(path_base+'.txt')
    obj.meta = meta
    channels = sorted([k for k in meta.keys() if 'Channel' in k])
    dye_names = [meta[c]['Dye Name'][0] for c in channels]
    for stream,dye in zip(obj.stacks,dye_names):
        stream.meta = meta.copy()
        stream.meta['channel'] = dye
    obj.order_stacks(dye_order)
    return obj





def _is_glob_or_names(path):
    "return True if path is a glob-like string or a collection of strings"
    if isinstance(path, str):
        return '*' in path
    else:
        return bool(iterable(path))

def from_any(path, *args, **kwargs):
    """Dispatch to an appropriate class constructor depending on the file name
    Should cover aroun 90% of usecases. For the remaining 10%, use direct constructors
    """
    images =  ('bmp', 'jpg', 'jpeg', 'png', 'tif','tiff', 'ppm', 'pgm')
    if isinstance(path, (FrameStackMono, FStackColl)):
        return path
    if isinstance(path, np.ndarray):
        handler =  from_array
    elif _is_glob_or_names(path):
        handler = from_images
        xml_try = leica.get_xmljob(path)
        if xml_try and not 'flavor' in kwargs:
            kwargs['flavor'] = 'leica'
    elif isinstance(path, str):
        _,ending = os.path.splitext(path)
        ending = ending.lower()[1:]
        handler_dict = {
        'npy': from_npy,
        'h5':  from_h5,
        'tif': from_tiff,
        'tiff':from_tiff,
        'mlf': from_plsi,
        'pls': from_plsi,
        'mes': from_mes,
        }
        if not ending in handler_dict:
            raise TypeError("Can't dispatch the file with ending %s \
            onto a supported FrameStack format"%ending)
        handler = handler_dict[ending]
    else:
        raise TypeError("Can't dispatch the `path` onto a supported FrameStack format")
    obj =  handler(path, *args, **kwargs)
    return obj


def valid_kwargs(handlerClass,kwargs):
    spec = inspect.getargspec(handlerClass.__init__)
    kwargs = {k:kwargs[k] for k in kwargs if k in spec[0]}
    return kwargs

def construct_with_kwargs(handlerClass, *args, **kwargs):
    return handlerClass(*args,**valid_kwargs(handlerClass, kwargs))


def _open_seq(path, *args, **kwargs):
    """
    **  Deprecated and broken **
    Dispatch to an appropriate class constructor depending on the file name

    Parameters:
      - path: (`str`) -- path to load data from. Can be a glob-style pattern or
        a single file name.
      - `*args`, `**kwargs`: will be dispatched to the actual class' `__init__` call

    Returns:
      - `instance`  of an appropriate Frame Sequence class
    """
    images =  ('bmp', 'jpg', 'jpeg', 'png', 'tif','tiff', 'ppm', 'pgm')
    if isinstance(path, np.ndarray):
        return FStackM_arr(path, *args, **kwargs)
    if isinstance(path, FrameStackMono):
        return path
    #ending = re.findall('[^*\.]+', path)[-1].lower()
    _,ending = os.path.splitext(path).lower()
    if ending == 'txt':
        handler = FStackM_txt
    elif ending == 'mes':
        handler = FStackM_mes
    elif ending in ('mlf', 'pls'):
        handler = FStackM_plsi
    elif ending == 'npy':
        handler =  FStackM_npy
    elif ending == 'h5':
        handler = FStackM_hdf5
    elif ending in images:  # A collection of images or a big tiff
        if '*' in path: # many files
            from imfun import leica
            xml_try = leica.get_xmljob(path)
            print('*******', path,     xml_try)
            if 'xmlname' in kwargs or xml_try:
                handler =  FStackM_imgleic
            else:
                handler =  FStackM_img
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

    class FStackM_hdf5(FStackM_arr):
        "Base class for hdf5 files"
        def __init__(self, fname, dataset=None,**kwargs):
            parent = super(FStackM_hdf5, self)
            f = h5py.File(fname, 'r')


            if dataset and dataset not in f:
                print("Dataset name doesn't exist in file, setting to None ")
                print("The file %s has the following data sets:"%fname, ', '.join(f.keys()))                
                dataset = None

            if dataset is None: # no dataset name is provided
                keys = list(f.keys())
                if len(keys) == 1: # there is only one dataset, use it
                    dataset = keys[0]
                else:
                    raise KeyError("No or wrong dataset name provided and the file has\
                    more than one")
            arr = f[dataset]
            parent.__init__(arr,**kwargs)
            self.h5file = f # just in case we need it later

    class FStackColl_hdf5(FStackColl):
        "Base class for hdf5 files"
        def __init__(self, fname, **kwargs):
            parent = super(FStackColl_hdf5, self)            
            f = h5py.File(fname, 'r')
            datasets = list(f.keys())
            print("The file %s has the following data sets:"%fname, ', '.join(datasets))
            self.h5file = f # just in case we need it later
            stacks = []
            for dset in datasets:
                arr = f[dset]
                fsm = FStackM_arr(arr,**kwargs)
                fsm.meta['channel'] = dset
                stacks.append(fsm)
            parent.__init__(stacks)
                

    class FStackM_hdf5_lsc(FStackM_arr):
        "Class for hdf5 files written by pylsi software"
        def __init__(self, fname, frame_filters = None):
            parent = super(FStackM_hdf5_lsc, self)
            #parent.__init__()
            f = h5py.File(fname, 'r')
            t = f['tstamps']
            self.tv = (t-t[0])/1e6 # relative time, in s
            dt = np.median(np.diff(self.tv))
            meta = {'axes': core.units.alist_to_scale([dt,'sec'])}
            self.h5file = f # just in case we need it later
            parent.__init__(f['lsc'], frame_filters=frame_filters,meta=meta)

    def fstack_to_h5(seqlist, name, compress_level=-1,verbose=False):
        # todo: add metadata, such as time and spatial scales
        if os.path.exists(name):
            if verbose:
                sys.stderr.write("File exists, removing\n")
            os.remove(name)
        fid = h5py.File(name, 'w')
        L = np.sum(list(map(len, seqlist)))
        sh = np.min([s.frame_shape for s in seqlist], axis=0)
        print('shape:', sh)
        chunkshape = tuple([1] + list(sh))
        fullshape = tuple([L] + list(sh))
        if verbose:
            print("Full shape:", fullshape)
            print("Chunk shape:", chunkshape)
        kwargs = dict()
        if compress_level > -1:
            kwargs['compression'] = 'gzip'
            kwargs['compression_opts'] = compress_level

        dset = fid.create_dataset('data', fullshape, dtype=seqlist[0][0].dtype,
                                  chunks = chunkshape, **kwargs)
        k = 0
        for seq in seqlist:
            for f in seq:
                dset[k,...] = f
                if verbose:
                    sys.stderr.write('\r writing frame %02d out of %03d'%(k,L))
                k += 1
        fid.close()
        return name


except ImportError as e: # couln't import h5py
    print("Import Error", e)




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
        fs.meta['axes'][0] = QS(1./framerate, 's')
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
    print("Can't load OpenCV python bindings", e)

from . import cluster 
from .components import pca
def frame_exemplars_pca_som(fs, pcf=None, npc=None, som_gridshape=None):
    frames = fs[:]
    if npc is None:
        npc = 3+len(frames)//100
    if pcf is None: # no PCA_frames instance provided
        pcf = pca.PCA_frames(frames,npc=npc)
    coords = np.array([pcf.project(f) for f in frames])
    npc = min(npc, pcf.npc)
    if som_gridshape is None:
        som_gridshape = (1+len(frames)//100,1)
    som_result = cluster.som(coords[:,:npc], gridshape=som_gridshape)
    som_result = cluster.sort_clusters_by_size(som_result)
    centroids = (coords[som_result==_k].mean(axis=0) for _k in np.unique(som_result))
    exemplars = list(map(pcf.rec_from_coefs, centroids))
    return exemplars, som_result
    

        
