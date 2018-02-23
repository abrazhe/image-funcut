import numpy as np
#import time, sys
#from swan import pycwt

import itertools as itt
import operator as op
from functools import partial

from scipy import ndimage

from . import core
from .core import ifnot, coords
from functools import reduce

def locations(shape):
    """locations from a given shape (Cartesian product) as an iterator"""
    return itt.product(*map(range, shape))


def adaptive_threshold(arr, n = 3, k = 0,binfn=np.mean):
    """Adaptive-thresholded binarization for an array.

    Parameters:
      - `arr`: input array
      - `n`: (`int`) -- window half-size
      - `k`: (`number`) -- threshold will be at mean(window)-k

    Returns:
      - mask: (array) -- 1 where array > threshold, 0 otherwise
    """
    nrows,ncols = arr.shape
    out = np.zeros(arr.shape)
    for row,col in locations(arr.shape):
        sl = (slice((row-n)%nrows,(row+n+1)%nrows),
              slice((col-n)%ncols,(col+n+1)%ncols))
        #m = np.mean(arr[sl])
        m = binfn(arr[sl])
        if arr[row,col] > m - k:
            out[row,col] = 1
    return out


def auto_threshold(arr, init_th = None, max_iter = 1e7):
    """
    Automatic threhold with INTERMEANS(I) algorithm

    Parameters:
      - `arr`: array-like
      - `init_th`: starting threshold
      - `max_iter`: upper limit of iterations

    Returns:
      - threshold: float

    Based on:
    T. Ridler and S. Calvard, "Picture thresholding using an iterative
    selection method," IEEE Trans. Systems Man Cybernet., vol. 8, pp. 630-632,
    1978.
    """
    thprev = ifnot(init_th, np.median(arr))
    for i in range(int(max_iter)):
        ab = np.mean(arr[np.where(arr <= thprev)])
        av = np.mean(arr[np.where(arr > thprev)])
        thnext = 0.5*(ab+av)
        if thnext <= thprev:
                break
        thprev = thnext
    return thnext


def contiguous_regions(binarr):
    """
    Given a binary Nd array, return a sorted (by size) list of contiguous
    regions (True everywhere)
    Version without recursion. Relies on scipy.ndimage.find_objects
    """
    sh = binarr.shape
    regions = [[]]
    visited = np.zeros(sh, bool)
    N = np.prod(sh)

    regions = []
    labels, nlab = ndimage.label(binarr)
    for j, o in enumerate(ndimage.find_objects(labels)):
        #sys.stderr.write('\rlocation %06d out of %06d'%(j+1, nlab))
        origin =  np.asarray([x.start for x in o])
        #x1 = np.asarray(np.where(labels[o] == j+1)).T
        x1 = np.argwhere(labels[o] == j+1)
        regions.append( list(map(tuple, (x1 + origin))))

    regions.sort(key = lambda x: len(x), reverse=True)
    return [RegionND(x, binarr.shape) for x in regions]


def neighbours_x(loc,shape):
    """Return list of ajacent locations for a n-dimensional location

    Parameters:
      - loc: (`tuple`) -- location
      - shape: (`tuple`) -- shape of enclosing array

    Returns:
      - list of `tuple`s
    """
    n = len(loc)
    d = np.diag(np.ones(n))
    x = np.concatenate((d,-d)) + loc
    return list(filter(partial(valid_loc, shape=shape), list(map(tuple, x))))


def neighbours_2(loc, shape):
    """Return list of adjacent locations"""
    r,c = loc
    return [x for x in ((r,c+1), (r,c-1),
                   (r+1,c), (r-1,c),
                   (r-1,c-1), (r+1,c-1),
                   (r-1, c+1), (r+1,c+1)) if valid_loc(x, shape)]

neighbours = neighbours_x

def valid_loc(loc,shape):
    "Test if location not outside bounds"
    return reduce(op.__and__, [(0 <= x < s) for x,s in zip(loc,shape)])


def filter_density(mask, rad=3, size=5, fn = lambda m,i,j: m[i,j]):
    """Density-based filter a binary mask.

    Pixel is `True` if there are more than `size` pixels within a radius of
    `rad` from the pixel

    Parameters:
      - `mask`: binary mask
      - `rad` : neighborhood radius
      - `size`: number of pixels required to be within the neighborhood
      - `fn`  : getter function, ``lambda m,i,j: m[i,j]`` by default

    Returns:
      - filtered mask
    """
    rows, cols = mask.shape
    X,Y = np.meshgrid(range(cols), range(rows))
    in_circle = coords.in_circle
    out = np.zeros((rows,cols), np.bool)
    for row,col in locations(mask.shape):
        if fn(mask,row,col):
            a = in_circle((col,row),rad)
            if np.sum(mask*a(X,Y))>size:
                out[row,col] = True
    return out

def majority(bimage, th = 5, mod = False):
    """Perform majority operation on the input binary image"""
    rows, cols = bimage.shape
    out = np.zeros((rows,cols), np.bool)
    for row in range(1,rows):
        for col in range(1,cols):
            sl = (slice(row-1,row+2), slice(col-1,col+2))
            x = np.sum(bimage[sl])
            out[(row,col)] = (x >= th)
            if mod:
               out[(row,col)] *= bimage[row,col]
    return out


def filter_mask(mask, fn, args=()):
    """Split a mask into contiguous regions,
    filter them by a provided function, and return result as a mask
    """
    regs = contiguous_regions(mask)
    filtered_regs = fn(regs, *args)
    z = np.zeros(mask.shape, dtype=np.bool)
    if len(filtered_regs) >1:
        return reduce(lambda a,b:a+b,
                      [z]+[r.tomask() for r in filtered_regs])
    else:
        return z

def filter_size_regions(regions, min_size=5):
    """Filter contiguous regions (clusters) by their size"""
    return [r for r in regions if r.size()>min_size]

def filter_shape_regions(regions, th = 2):
    """Filter contiguous regions (clusters) by circularity of shape"""
    return [r for r in regions
            if (r.linsize() > th*np.sqrt(r.size()))]

def glue_adjacent_regions(regions, max_distance=10):
    """Go through a sequence of regions, for each pair of closely-spaced
    regions, unite the to make a single region. Return the new list"""
    L = len(regions)
    acc = []
    def _glue_if(r1,r2):
        if distance_regions(r1,r2) < max_distance:
            return unite_2regions(r1,r2)
        else:
            return None
    def _loop(regs):
        if len(regs) == 1:
            acc.append(regs[0]); return
        if len(regs) < 1: return
        first,rest = regs[0], regs[1:]
        x = [_f for _f in [_glue_if(first,x) for x in rest] if _f]
        if x == []:
            acc.append(first)
            _loop(rest)
        else:
            a = reduce(_glue_if, x)
            _loop([a] + [b for b in rest if not regions_overlap(a,b)])

    _loop(regions)
    return acc

def regions_overlap(r1,r2):
    """Test if two regions overlap"""
    x = False
    for loc in r1.locs:
        if loc in r2.locs:
            return True
    return False

def unite_2regions(region1,region2):
    "Glue together two regions"
    return RegionND(list(region1.locs) + list(region2.locs), region1.shape)
    return


def distance_regions(r1, r2, fn=min, start=1e9):
    """Operate on pairwise distances between two regions
    if fn is =min=, smallest distance is returned,
    if fn is =max=, largest distance is returned
    """
    dists = [coords.eu_dist(*pair) for pair in
             itt.product(r1.borders(), r2.borders())]
    return reduce(fn, dists, start)


def distance_regions_centra(r1,r2):
    """Return distance between centroids of the two regions"""
    return coords.eu_dist(r1.center(), r2.center())

class RegionND:
    "Basic class for a contiguous region. Can make masks from it"
    def __init__(self, locs, shape):
        #self.locs = [loc[::-1] for loc in locs]
        self.locs = locs
        self.shape = shape # shape of containing array
    def ax_extent(self,axis=0):
        values = [x[axis] for x in self.locs]
        return np.max(values)-np.min(values)
    def size(self,):
        return len(self.locs)
    def center(self):
        return np.mean(self.locs,0)
    def borders(self):
        return (l for l in self.locs if
                l[0] == 0 or l[1] == 0 or l[0]==self.shape[0] or l[1] == self.shape[1]
                or 
                len([x for x in neighbours(l,self.shape) if x not in self.locs]))
    def linsize(self,):
        dists = [coords.eu_dist(*pair) for pair in core.misc.allpairs(self.borders())]
        return reduce(max, dists, 0)

        pass
    def tomask(self):
        m = np.zeros(self.shape, bool)
        for loc in self.locs: m[loc]=True
        return m


#----------------------------
