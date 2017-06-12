 # a/b will always return float

import numpy as np

from .constants import *

_maxshape=100e6

def memsafe_arr(shape, dtype=_dtype_):
    import tempfile as tmpf
    shape = tuple(shape)
    N = np.prod(shape)
    if N < _maxshape_:
        return np.zeros(shape, dtype=dtype)
    else:
        print("Using memory-mapped arrays...")
        _tmpfile = tmpf.TemporaryFile()
        out = np.memmap(_tmpfile, dtype=dtype, shape=shape)
        _tmpfile.close()
    return out
    


def ravel_frames(frames):
    #l,w,h = frames.shape
    if isinstance(frames, np.ndarray):
        l,w,h = frames.shape
        return frames.reshape(l,w*h)
    else:
        return np.array([np.ravel(f) for f in frames])

def shape_frames(X, shape):
    (nrows,ncols) = shape
    if isinstance(X, np.ndarray):
        Nt,Np = X.shape
        return X.reshape(Nt, nrows, ncols)
    else:
        return np.array([_x.reshape(nrows,ncols) for _x in X])
    



def embedding(arr, delarrp=True):
    """Return an *embeding* of the non-zero portion of an array.

    Parameters:
      - `arr`: array
      - `delarrp`: predicate whether to delete the `arr`

    Returns tuple ``(out, (sh, slices))`` of:
        * out: array, which is a bounding box around non-zero elements of an input
          array
        * sh:  full shape of the input data
        * slices: a list of slices which define the bounding box
    
    
    """
    sh = arr.shape
    b = np.argwhere(arr)
    starts, stops = b.min(0), b.max(0)+1
    slices = [slice(*p) for p in zip(starts, stops)]
    out =  arr[slices].copy()
    if delarrp: del arr
    return out, (sh, slices)

def embedded_to_full(x):
    """Restore 'full' object from it's *embedding*, 
    e.g. full image from  object subframe
    """
    data, (shape, xslice) = x
    out = np.zeros(shape, _dtype_)
    out[xslice] = data
    return out
    




def clip_and_rescale(arr,nout=100):
    "convert data to floats in 0...1, throwing out nout max values"
    #out = arr - np.min(arr)
    cutoff = 100*(1-float(nout)/np.prod(arr.shape))
    m = np.percentile(arr, cutoff)
    return np.where(arr < m, arr, m)/m

def rescale(arr):
    "Rescales array to [0..1] interval"
    out = arr - np.min(arr)
    return out/np.max(out)
