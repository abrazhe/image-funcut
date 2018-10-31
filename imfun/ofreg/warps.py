
import itertools as itt


import numpy as np

from scipy.ndimage.interpolation import map_coordinates

from imfun import fseq
from ..core import fnutils

import pickle
import gzip as gz
import collections

_boundary_mode = 'nearest'


class Warp:
    def __init__(self):
        self.field_ = None
        self.fn_ = None
    @staticmethod
    def from_array(flow_field):
        o = Warp()
        o.field_ = np.asarray(flow_field)
        o.sh = flow_field[0].shape
        return o
    @staticmethod
    def from_function(fn,sh):
        o = Warp()
        o.fn_ = fn
        o.sh_ = sh
        return o
    @property
    def field(self):
        if self.fn_ is None and self.field_ is None:
            raise ValueError("Neither warp function nor flow field defined")
        if self.field_ is None:
            self.field_ = flow_from_fn(self.fn_,self.sh_)
        return self.field_
    def __add__(self, other):
        if self.fn_ and other.fn_:
            return Warp.from_function(fnutils.flcompose2(self.fn_, other.fn_),self.sh_)
        else:
            return Warp.from_array(self.field + other.field)
    def __call__(self, img,mode=_boundary_mode):
        sh = img.shape
        if np.ndim(img) == 2:
            start_coordinates = np.meshgrid(*list(map(np.arange, sh[:2][::-1])))
            if self.fn_ is not None:
                new_coordinates = self.fn_(start_coordinates)
            else:
                new_coordinates = [c+f for c,f in zip(start_coordinates, self.field)]
            return map_coordinates(img, new_coordinates[::-1], mode=mode)
        elif np.ndim(img) > 2:
            return np.dstack([self(img[...,c],mode) for c in range(img.shape[-1])])
        else:
            raise ValueError("Can't handle image of such shape: {}".format(sh))
        
def flow_from_fn(fn, sh):
    start_coordinates = np.meshgrid(*list(map(np.arange, sh[::-1])))
    return fn(start_coordinates)-np.array(start_coordinates)

#def fn_from_flow(flow):
#    def _regfn(coordinates):
#        return [c + p for c,p in zip(coordinates, flow)]
#    return _refn

## Inventing functions like these makes me think I should make warp a Class
## or a tree of classes which should take care of  adding together via __add__

import operator as op
def compose_warps(*warps):
    _pcalls = list(map(callable, warps))
    if np.all(_pcalls):
        return fnutils.flcompose(*warps)
    else:
        #def _fnconv(w):
        #    if callable(w): return flow_from_fn(w)
        #    else: return w
        return Warp.from_array(np.sum((isinstance(w, collections.Callable) and flow_from_fn(w) or w for w in warps),axis=0))


# def apply_warp(warp, img ,mode=_boundary_mode):
#     """Given an image and a function to warp coordinates,
#     or a pair (u,v) of horizontal and vertical flows
#     warp image to the new coordinates.
#     In case of a multicolor image, run this function for each color"""
#     sh = img.shape
#     if np.ndim(img) == 2:
#         start_coordinates = np.meshgrid(*map(np.arange, sh[:2][::-1]))
#         if callable(warp):
#             new_coordinates = warp(start_coordinates)
#         elif isinstance(warp, (tuple, list, np.ndarray)):
#             new_coordinates = [c+f for c,f in zip(start_coordinates, warp)]
#         else:
#             raise ValueError("warp can be either a function or an array")
#         return map_coordinates(img, new_coordinates[::-1], mode=mode)
#     elif np.ndim(img) > 2:
#         return np.dstack([apply_warp(warp,img[...,c],mode) for c in range(img.shape[-1])])
#     else:
#         raise ValueError("Can't handle image of such shape: {}".format(sh))


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
    print("Can't load `dill` package, won't be able to save warps as functions")
    print("""Consider installing it by one of the following commands:
> pip install https://github.com/uqfoundation/dill/archive/master.zip
OR
> conda install dill
""")

try:
    from pathos.pools import ProcessPool
    _with_pathos_ = True
except ImportError:
    print("""Can't load `pathos` package, parallel maps won't work.
Consider installing it by one of the following commands:
> pip install git+https://github.com/uqfoundation/pathos
OR
> pip install https://github.com/uqfoundation/pathos/archive/master.zip
""")

def to_pickle(name,warps):
    with gzip.open(name, 'wb') as recipe:
        pickle.dump(warps, recipe)


def from_pickle(name):
    with gzip.open(name, 'rb') as recipe:
        return pickle.load(recipe)

def to_npy(name, warps):
    np.save(name, warps)

def from_npy(name):
    return np.load(name)

def to_dct_encoded(name, warps, s=5, upto=100, th=3):
    """Convert warps to DCT coefficients and save to a .npy file
    """
    codes = (dct_encode(w.field, s, upto, th)[0] for w in warps)
    sparse_codes = [[sparse.coo_matrix(f) for f in cw] for cw in codes]
    to_npy(name, sparse_codes)

def from_dct_encoded(name, **fnargs):
    codes = from_npy(name)
    return [Warp.from_array(dct_decode(c.toarray())) for c in codes]
    

def _to_dct_encoded_old(name, warps, upto=50, th=3):
    """Convert warps to DCT coefficients and save to a .npy file
    """
    w = warps[0]
    sh = w.field.shape[1:]
    D = make_dct_dict(w.field[0].shape, upto)
    Dr = D.reshape(len(D),-1)
    wwx = np.array([w.field for w in warps]).reshape(-1,2,np.prod(sh))
    xflows,yflows = np.split(wwx, 2, axis=1)
    xcodes,ycodes = (np.squeeze(fl).dot(Dr.T) for fl in (xflows,yflows))
    del wwx, xflows, yflows
    to_npy(name, [(np.concatenate((xc,yc)),sh) for xc,yc in zip(xcodes, ycodes)])
    #to_npy(name, [dct_encode(w.field, upto, D) for w in warps])

def _from_dct_encoded_old(name, **fnargs):
    """Load DCT codes from an .npy file and convert to warp objects
    """
    codes = from_npy(name)
    cf,sh = codes[0]
    D = make_dct_dict(sh,int(np.sqrt(len(cf)//2)))
    Dr = D.reshape(len(D),-1)
    cx = np.array([c[0] for c in codes])
    xflows = cx[:,:len(D)].dot(Dr).reshape((-1,)+sh)
    yflows = cx[:,len(D):].dot(Dr).reshape((-1,)+sh)
    return list(map(Warp.from_array, ((u,v) for u,v in zip(xflows, yflows))))
    #return map(Warp.from_array, (dct_decode(c,D) for c in  codes))


def map_warps(warps, frames, njobs=4):
    """
    returns result of applying warps for given frames (one warp per frame)
    """
    if njobs > 1 and _with_pathos_:
        pool = ProcessPool(nodes=njobs)
        #out = pool.map(apply_warp, warps, frames)
        out = pool.map(Warp.__call__, warps, frames,chunksize=1)
        #pool.close()
        out = np.array(out)
    else:
        out = np.array([w(f) for w,f in zip(warps, frames)])
    if isinstance(frames, (fseq.FrameStackMono, fseq.FStackColl)):
        out = fseq.from_array(out)
        out.meta = frames.meta.copy()
        if isinstance(frames, fseq.FStackColl):
            for s1,s2 in zip(out.stacks, frames.stacks):
                s1.meta = s2.meta.copy()
    return out



from ..filt import dct2d, idct2d
from ..filt.dctsplines import l2spline_thresholded

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
    return np.array(dct_dict)


from scipy import sparse
def dct_encode(flow, s=5, upto=50, th=0.5):
    coefs = np.array([l2spline_thresholded(f,s,nharmonics=upto,th=th) for f in flow])
    return coefs


def dct_decode(coefs):
    return np.array([idct2d(c) for c in coefs])
    
    

def _dct_encode_old(flow,upto=20,D=None):
    if D is None:
        D = make_dct_dict(flow[0].shape, upto)
    coefs = np.concatenate([[np.sum(d*a) for d in D] for a in flow])
    return coefs, flow[0].shape

def _dct_decode_old(coefs,shape,D=None):
    upto = int(np.sqrt(len(coefs)//2))
    if D is None:
        D = make_dct_dict(shape, upto=upto)
    Ld = len(D)
    xflow = np.sum([k*el for k,el in zip(coefs[:Ld],D)],0)
    yflow = np.sum([k*el for k,el in zip(coefs[Ld:],D)],0)
    return np.array([xflow,yflow])


def with_encoding(wf, harmonics=20, img_shape=None,D=None):
    """
    if warps are functions, must provide `img_shape` argument
    """
    def _ef(*args, **kwargs):
        warp = wf(*args, **kwargs)
        return dct_encode(warp.field, harmonics,D)
    return _ef
