
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


def default_kernel(self):
    """
    Default kernel for aliased_pix_iter2
    Used in 2D convolution of each frame in the sequence
    """
    kern = ones((3,3))
    kern[1,1] = 2.0
    return kern/sum(kern)


class FrameSequence():
    def mask_reduce(self,mask):
        "create 1D vector from mask (or slice)"
        return np.asarray([np.mean(f[mask]) for f in self.frames()])

    def frame_slices(self, sliceobj):
        "return iterator over subframes"
        return (f[sliceobj] for f in self.frames())

    def mean_frame(self,):
        frameit = itt.imap(np.float64, self.frames())
        res = frameit.next()
        for k,frame in enumerate(frameit):
            res += frame
        return res/(k+2)

    def aslist(self, maxN = None, fn = lambda x: x):
        return list(itt.islice(itt.imap(fn, self.frames()), maxN))

    def as3darray(self, maxN = None, fn = lambda x: x):
        return np.asarray(self.aslist(maxN, fn))
    
    def length(self):
        for k,_ in enumerate(self.frames()):
            pass
        return k+1

    def simple_pix_iter(self, maxN, fn = lambda x:x):
        arr = self.as3darray(maxN,fn)
        nrows, ncols = arr.shape[1:]
        for row in range(nrows):
            for column in range(columns):
                yield arr[:,i,j], i, j
    def conv_pix_iter(self, kern = default_kernel()):
        fn = lambda a: signal.convolve2d(a, kern)[1:-1,1:-1]
        return simple_pix_iter(self, maxN, fn)

    def shape(self,):
        return self.frames().next().shape

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


