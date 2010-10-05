
# Classes for sequencies of frames

import glob
import itertools as itt
import numpy as np
import tempfile as tmpf
from scipy import signal

from matplotlib.pyplot import imread

from imfun import lib
ifnot = lib.ifnot

_maxshape_ = 1e5

def sorted_file_names(pat):
    "Returns a sorted list of file names matching a pattern"
    x = glob.glob(pat)
    x.sort()
    return x

def iter_files(pattern, loadfn):
    "From a pattern, return iterator over data sequence"
    return itt.imap(loadfn, sorted_file_names(pattern))


def fseq_from_glob(pattern, ch=None, loadfn=np.load):
    "Sequence of frames from filenames matching a glob"
    if ch is None:
        return iter_files(pattern, loadfn)
    else:
        if pattern[-3:] == 'png':
            getter = lambda frame: frame[::-1,:,ch]
        else:
            getter = lambda frame: frame[:,:,ch]
        return  itt.imap(getter, iter_files(pattern, loadfn))


## def default_kernel():
##     """
##     Default kernel for conv_pix_iter
##     Used in 2D convolution of each frame in the sequence
##     """
##     kern = np.ones((3,3))
##     kern[1,1] = 4.0
##     return kern/kern.sum()


def fseq_to_mpg(seq, name, fps = 25, **kwargs):
    "Creates an mpg  movie from frame sequence with mencoder"
    import os, sys
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    vmin = np.min(map(np.min, seq.frames())) # for scale
    vmax = np.max(map(np.max, seq.frames())) # for scale
    L = seq.length()
    for i,frame in enumerate(seq.frames()):
        ax.cla()
        ax.imshow(frame, aspect='equal', vmin=vmin, vmax=vmax, **kwargs)
        fname = '_tmp%06d.png'%i
        fig.savefig(fname)
        sys.stderr.write('\r saving frame %06d of %06d'%(i+1, L))
    print 'Running mencoder, this can take a while'
    os.system("mencoder 'mf://_tmp*.png' -mf type=png:fps=%d -ovc lavc -lavcopts vcodec=wmv2 -oac copy -o %s.mpg"%(fps,name))
    os.system("rm -f _tmp*.png")
    

def gauss_kern(xsize=1.5, ysize=None):
    """ Returns a normalized 2D gauss kernel for convolutions """
    xsize = int(xsize)
    ysize = ysize and int(ysize) or xsize
    x, y = np.mgrid[-xsize:xsize+1, -ysize:ysize+1]
    g = np.exp(-(x**2/float(xsize) + y**2/float(ysize)))
    return g / g.sum()


default_kernel = gauss_kern


def gauss_blur(X,size=1.0):
    return signal.convolve2d(X,gauss_kern(size),'same')

class FrameSequence:
    "Base class for sequence of frames"
    def get_scale(self):
        if hasattr(self, '_scale_set'):
            scale_flag = self._scale_set
            dx,dy, scale_flag = self.dx, self.dy, self._scale_set
        else:
            dx,dy,scale_flag = 1,1,None
        return dx, dy, scale_flag

    def set_scale(self, dx=None, dy=None):
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

    def timevec(self,):
        "vector of time values"
        L = self.length()
        dt = self.dt
        return np.arange(0, (L+2)*dt, dt)[:L]

    def mask_reduce(self,mask):
        "create 1D vector from mask (or slice)"
        return np.asarray([np.mean(f[mask]) for f in self.frames()])

    def frame_slices(self, sliceobj,fn=None):
        "iterator over subframes"
        if sliceobj:
            return (f[sliceobj] for f in self.frames(fn))
        else:
            return self.frames(fn)

    def mean_frame(self,nframes = None, fn=None):
        "Create mean image over N frames (all by default)"
        L = self.length()
        frameit = itt.imap(np.float64, self.frames(fn))
        res = np.copy(frameit.next())
        nframes = min(ifnot(nframes,L), L)
        for k,frame in enumerate(frameit):
            res += frame
            if k >= nframes:
                break
        return res/(k+2)

    def aslist(self, fn = None, maxN=None, sliceobj=None):
        "returns a list of frames"
        return list(self.asiter(maxN,fn,sliceobj))

    def asiter(self, fn = None, maxN=None, sliceobj=None):
        "returns a modified iterator over frames"
        fiter = self.frame_slices(sliceobj, fn)
        return itt.islice(fiter, maxN)

    def as3darray(self, fn = None, maxN = None, sliceobj=None):
        fiter = self.frame_slices(sliceobj, fn=fn)
        shape = self.shape(sliceobj)
        N =  self.length()*shape[0]*shape[1]
        ## If total size of data is les than _maxshape_ use normal arrays,
        ## otherwise use memory-mapped arrays
        if N < _maxshape_:
            out = np.zeros((self.length(), shape[0], shape[1]))
        else:
            _tmpfile = tmpf.TemporaryFile('w+',dir='/tmp/')
            out = np.memmap(_tmpfile, dtype=np.float64,
                            shape=(self.length(), shape[0], shape[1]))
        for k,frame in enumerate(itt.islice(fiter, maxN)):
            out[k,:,:] = frame
        if hasattr (out, 'flush'):
            out.flush()
        return out
        #return np.asarray(self.aslist(*args, **kwargs))


    def as_memmap_array(self,  fn = None, maxN = None, sliceobj=None):
        fiter = self.frame_slices(sliceobj, fn)
        shape = self.shape(sliceobj)
        _tmpfile = tmpf.TemporaryFile('w+')
        out = np.memmap(_tmpfile, dtype=np.float64,
                        shape=(self.length(), shape[0], shape[1]))
        for k,frame in enumerate(itt.islice(fiter, maxN)):
            out[k,:,:] = frame
        out.flush()
        return out
        
    def pix_iter(self, mask=None, maxN=None, rand=False, **kwargs):
        "Iterator over time signals from each pixel"
        arr = self.as3darray(maxN, **kwargs)
        if mask== None:
            mask = np.ones(self.shape(), np.bool)
        nrows, ncols = arr.shape[1:]
        rcpairs = [(r,c) for r in xrange(nrows) for c in xrange(ncols)]
        if rand: rcpairs = np.random.permutation(rcpairs)
        for row,col in rcpairs:
            if mask[row,col]:
                ## asarray to convert from memory-mapped array
                yield np.asarray(arr[:,row,col]), row, col

    def length(self):
        if not hasattr(self,'_length'):
            k = 0
            for _ in self.frames():
                k+=1
            self._length = k
            return k
        else:
            return self._length

    def shape(self,sliceobj=None):
        return self.frame_slices(sliceobj).next().shape

    def pw_transform(self, pwfn,**kwargs):
        """Create another frame sequence, pixelwise applying a function"""
        nrows, ncols = self.shape()
        out = np.zeros((self.length(), nrows, ncols))
        for v, row, col in self.pix_iter(**kwargs):
            out[:,row,col] = pwfn(v)
        return FSeq_arr(out, dt = self.dt, dx=self.dx, dy = self.dy)
    def export_img(self, path, base = 'fseq-export-', figsize=(4,4),
                   format='.png', **kwargs):
        import  sys
        import matplotlib.pyplot as plt
        lib.ensure_dir(path)
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        vmin = np.min(map(np.min, self.frames())) # for scale
        vmax = np.max(map(np.max, self.frames())) # for scale
        L = self.length()
        for i,frame in enumerate(self.frames()):
            ax.cla()
            ax.imshow(frame, aspect='equal', vmin=vmin, vmax=vmax, **kwargs)
            fname =  path + base + '%06d.png'%i
            fig.savefig(fname)
            sys.stderr.write('\r saving frame %06d of %06d'%(i+1, L))
        plt.close()
    def export_mpeg(self, mpeg_name, fps = None, **kwargs):
        import os
        "Creates an mpg  movie from frame sequence with mencoder"
        print "Saving frames as png"
        path, base = './', '_tmp'
        if fps is None:
            fps = 1/self.dt
        self.export_img(path, base, **kwargs)
        print 'Running mencoder, this can take a while'
        mencoder_string = """mencoder mf://_tmp*.png -mf type=png:fps=%d\
        -ovc lavc -lavcopts vcodec=wmv2 -oac copy -o %s.mpg"""%(fps,mpeg_name)
        os.system(mencoder_string)
        fnames = (path + base + '%06d.png'%i for i in xrange(self.length()))
        map(os.remove, fnames)

class FSeq_arr(FrameSequence):
    def __init__(self, arr, dt = 1.0, fns = [],
                 dx = None, dy = None):
        self.dt = dt
        self.data = arr
        self.hooks = []
        self.fns = fns
        self.set_scale(dx, dy)
    def length(self):
        return self.data.shape[0]
    def frames(self, fn=None):
        #fn = ifnot(fn, self.fn)
        fn = ifnot(fn, lib.flcompose(identity, *self.fns))
        return itt.imap(fn, (frame for frame in self.data))


def identity(x):
    return x

class FSeq_glob(FrameSequence):
    def __init__(self, pattern, ch=0, dt = 1.0, fns = [],
                 dx = None, dy = None):
        self.pattern = pattern
        self.ch = ch
        self.dt = dt
        self.fns = fns
        self.set_scale(dx, dy)
            
    def frames(self, fn = None):
        fn = ifnot(fn, lib.flcompose(identity, *self.fns))
        ## Examples of processing functions can be found in scipy.ndimage module
        ## TODO: a list of hook functions
        return itt.imap(fn, fseq_from_glob(self.pattern, self.ch, self.loadfn))

class FSeq_img(FSeq_glob):
    loadfn = lambda self,y: imread(y)

class FSeq_txt(FSeq_glob):
    loadfn= lambda self,y: np.loadtxt(y)

class FSeq_npy(FSeq_glob):
    loadfn= lambda self,y: np.load(y)

class FSeq_imgleic(FSeq_img):
    def __init__(self, pattern, ch=0, fns=[]):
        FSeq_glob.__init__(self, pattern,ch=ch)
        self.fns = []
        try:
            from imfun import leica
            self.lp = leica.LeicaProps(self.pattern.split('*')[0])
            self.dt = self.lp.dt # sec
            self.set_scale(self.lp.dx, self.lp.dy) # um/pix
        except Exception as e:
            print "Got exception, ", e
            pass


from imfun.MLFImage import MLF_Image

class FSeq_mlf(FrameSequence):
    "Class for MLF multi-frame images"
    def __init__(self, fname, fns = []):
        self.mlfimg = MLF_Image(fname)
        self.dt = self.mlfimg.dt/1000.0
        self.fns = []
        self.set_scale()
    def frames(self, fn = None):
        #fn = ifnot(fn,self.fn)
        fn = ifnot(fn, lib.flcompose(identity, *self.fns))        
        return itt.imap(fn,self.mlfimg.flux_frame_iter())
        #return itt.imap(lambda x: x[0], self.mlfimg.frame_iter())
#    def shape(self): # return it back afterwards
#        return self.mlfimg.ydim,self.mlfimg.xdim
    def length(self):
        return self.mlfimg.nframes

import PIL.Image as Image
import matplotlib.image as mpl_img
class FSeq_multiff(FrameSequence):
    "Class for multi-frame tiff files"
    def __init__(self, fname, dt=1.0):
        self.dt = dt
        self.im = Image.open(fname)
        self.set_scale()
    def frames(self):
        count = 0
        while True:
            try:
                self.im.seek(count)
                count += 1
                yield mpl_img.pil_to_array(self.im)
            except EOFError:
                break
            
