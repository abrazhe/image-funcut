
# Classes for sequencies of frames

import glob
import itertools as itt
import numpy as np
from scipy import signal

from pylab import imread


def sorted_file_names(pat):
    "Returns a sorted list of file names matching a pattern"
    x = glob.glob(pat)
    x.sort()
    return x

def iter_files(pattern, loadfn):
    "From a pattern, return iterator over data sequence"
    return itt.imap(loadfn, sorted_file_names(pattern))


def fseq_from_glob(pattern, ch=None, loadfn=np.load):
    if ch is None:
        return iter_files(pattern, loadfn)
    else:
        getter = lambda frame: frame[:,:,ch]
        return  itt.imap(getter, iter_files(pattern, loadfn))


def default_kernel():
    """
    Default kernel for conv_pix_iter
    Used in 2D convolution of each frame in the sequence
    """
    kern = np.ones((3,3))
    kern[1,1] = 2.0
    return kern/sum(kern)


class FrameSequence():
    def mask_reduce(self,mask):
        "create 1D vector from mask (or slice)"
        return np.asarray([np.mean(f[mask]) for f in self.frames()])

    def frame_slices(self, sliceobj):
        "return iterator over subframes"
        if sliceobj:
            return (f[sliceobj] for f in self.frames())
        else:
            return self.frames()

    def mean_frame(self,):
        frameit = itt.imap(np.float64, self.frames())
        res = frameit.next()
        for k,frame in enumerate(frameit):
            res += frame
        return res/(k+2)

    def aslist(self, maxN = None, fn = lambda x: x, sliceobj=None):
        "returns a list of frames"
        if sliceobj:
            fiter = self.frame_slices(sliceobj)
        else:
            fiter = self.frames()
        return list(itt.islice(itt.imap(fn, fiter), maxN))

    def as3darray(self, maxN = None, fn = lambda x: x, sliceobj=None):
        if sliceobj:
            fiter = self.frame_slices(sliceobj)
        else:
            fiter = self.frames()
        shape = self.shape(sliceobj)
        out = np.zeros((self.length(), shape[0], shape[1]))
        for k,frame in enumerate(itt.islice(itt.imap(fn, fiter), maxN)):
            out[k,:,:] = frame
        return out
        #return np.asarray(self.aslist(*args, **kwargs))
    
    def length(self):
        for k,_ in enumerate(self.frames()):
            pass
        return k+1

    def pix_iter(self, maxN=None, fn = lambda x:x, sliceobj=None):
        arr = self.as3darray(maxN,fn,sliceobj=sliceobj)
        nrows, ncols = arr.shape[1:]
        for row in range(nrows):
            for col in range(ncols):
                yield arr[:,row,col], row, col

    def conv_pix_iter(self, kern = default_kernel(),
                      *args, **kwargs):
                      #maxN=None, sliceobj=None):
        if kern is None: kern = default_kernel()
        def _fn(a):
            return signal.convolve2d(a, kern)[1:-1,1:-1]
        #fn = lambda a: signal.convolve2d(a, kern)[1:-1,1:-1]
        kwargs.update({'fn':_fn})
        return self.pix_iter(*args, **kwargs)

    def shape(self,sliceobj=None):
        return self.frame_slices(sliceobj).next().shape

class FSeq_glob(FrameSequence):
    def __init__(self, pattern, ch=0, dt = 1.0):
        self.pattern = pattern
        self.ch = ch
        self.dt = dt
    def frames(self):
        return fseq_from_glob(self.pattern, self.ch, self.loadfn)


class FSeq_img(FSeq_glob):
    loadfn = lambda self,y: imread(y)
        
class FSeq_txt(FSeq_glob):
    loadfn= lambda self,y: np.loadtxt(y)

class FSeq_npy(FSeq_glob):
    loadfn= lambda self,y: np.load(y)


from imfun.MLFImage import MLF_Image

class FSeq_mlf(FrameSequence):
    "Class for MLF multi-frame images"
    def __init__(self, fname):
        self.mlfimg = MLF_Image(fname)
        self.dt = self.mlfimg.dt/1000.0
    def frames(self):
        return itt.imap(lambda x: x[0], self.mlfimg.frame_iter())
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
    def frames(self):
        count = 0
        while True:
            try:
                self.im.seek(count)
                count += 1
                yield mpl_img.pil_to_array(self.im)
            except EOFError:
                break
            
