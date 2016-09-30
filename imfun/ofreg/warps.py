from __future__ import division
import itertools as itt


import numpy as np

from scipy.ndimage.interpolation import map_coordinates

from imfun import fseq

import pickle

_boundary_mode = 'constant'

def flow_from_fn(fn, sh):
    start_coordinates = np.meshgrid(*map(np.arange, sh[::-1]))
    return (fn(start_coordinates)-np.array(start_coordinates))

def fn_from_flow(flow):
    def _regfn(coordinates):
        return [c + p for c,p in zip(coordinates, flow)]
    return _refn

def apply_warp(warp, img ,mode=_boundary_mode):
    """Given an image and a function to warp coordinates,
    or a pair (u,v) of horizontal and vertical flows
    warp image to the new coordinates.
    In case of a multicolor image, run this function for each color"""
    sh = img.shape
    if np.ndim(img) == 2:
        start_coordinates = np.meshgrid(*map(np.arange, sh[:2][::-1]))
        if callable(warp):
            new_coordinates = warp(start_coordinates)
        elif isinstance(warp, (tuple, list, np.ndarray)):
            new_coordinates = [c+f for c,f in zip(start_coordinates, warp)]
        else:
            raise ValueError("warp can be either a function or an array")
        return map_coordinates(img, new_coordinates[::-1], mode=mode)
    elif np.ndim(img) > 2:
        return np.dstack([apply_warp(warp,img[...,c],mode) for c in range(img.shape[-1])])
    else:
        raise ValueError("Can't handle image of such shape: {}".format(sh))


_with_dill_ = False
_with_pathos_ = False
    

try:
    #!pip install https://github.com/uqfoundation/dill/archive/master.zip
    # OR
    #!conda install dill

    import dill
    _with_dill_ = True

    def from_dill(name):
        with open(name, 'rb') as recipe:
            return dill.load(recipe)
    def to_dill(name,warps):
        with open(name, 'wb') as recipe:
            dill.dump(warps, recipe)
except ImportError:
    print "Can't load `dill` package, won't be able to save warps as functions"
    print """Consider installing it by one of the following commands:
> pip install https://github.com/uqfoundation/dill/archive/master.zip
OR 
> conda install dill
"""
        
try:
    from pathos.pools import ProcessPool
    _with_pathos_ = True
except ImportError:
    print """Can't load `pathos` package, parallel maps won't work.
Consider installing it by one of the following commands:
> pip install git+https://github.com/uqfoundation/pathos
OR
> pip install https://github.com/uqfoundation/pathos/archive/master.zip
"""

def to_pickle(name,warps):
    with open(name, 'wb') as recipe:
        pickle.dump(warps, recipe)

    
def from_pickle(name):
    with open(name, 'rb') as recipe:
        return pickle.load(recipe)
    
def to_npy(name, warps):
    np.save(name, warps)

def from_npy(name):
    np.load(name)

def to_dct_encoded(name, warps, **fnargs):
    to_npy(name, [dct_encode(w, **fnargs) for w in warps])

def from_dct_encoded(name, **fnargs):
    codes = from_npy(name)
    return map(dct_decode, codes)
    
    
def map_warps(warps, frames, njobs=4):
    """
    returns result of applying warps for given frames (one warp per frame)
    """
    if njobs > 1 and _with_pathos_:
        pool = ProcessPool(nodes=njobs)
        out = pool.map(apply_warp, warps, frames)
        #pool.close()
        out = np.array(out)
    else:
        out = np.array([apply_warp(w,f) for w,f in itt.izip(warps, frames)])
    if isinstance(frames, (fseq.FrameStackMono, fseq.FStackColl)):
        out = fseq.from_array(out)
        out.meta = frames.meta
    return out
    


from scipy.fftpack import dct, idct

def dct2d(m,norm='ortho'):
    return dct(dct(m, norm=norm, axis=0),
               norm=norm, axis=1)

def idct2d(m,norm='ortho'):
    return idct(idct(m, norm=norm, axis=0),
               norm=norm, axis=1)

def make_dct_dict(shape, upto=20,dctnorm='ortho',maxnorm=False):
    dct_dict = []
    if upto is None: upto=size
    for i in range(upto):
        for j in range(upto):
            #if i + j > size: continue
            p = np.zeros(shape)
            p[i,j] = 1
            patch = idct2d(p,norm=dctnorm)
            if maxnorm:
                patch /= patch.max()
            dct_dict.append(patch)
    return dct_dict

def dct_encode(flow,upto=20,D=None):
    if D is None:
        D = make_dct_dict(flow[0].shape, upto)
    coefs = np.concatenate([[np.sum(d*a) for d in D] for a in flow])
    return coefs, flow[0].shape

def dct_decode((coefs, shape),D=None):
    upto = int(np.sqrt(len(coefs)//2))
    if D is None:
        D = make_dct_dict(shape, upto=upto)
    Ld = len(D)
    xflow = np.sum([k*el for k,el in itt.izip(coefs[:Ld],D)],0)
    yflow = np.sum([k*el for k,el in itt.izip(coefs[Ld:],D)],0)
    return np.array([xflow,yflow])


def with_encoding(wf, harmonics=20, img_shape=None,D=None):
    """
    if warps are functions, must provide `img_shape` argument
    """
    def _ef(*args, **kwargs):
        warp = wf(*args, **kwargs)
        if callable(warp):
            warp = flow_from_fn(warp,img_shape)
        return dct_encode(warp, harmonics,D)
    return _ef
