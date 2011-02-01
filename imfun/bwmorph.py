import numpy as np
import time, sys
#from imfun.aux_utils import ifnot
from swan import pycwt

import itertools as itt
import operator as op
from functools import partial

from scipy import ndimage

from imfun import lib, ui
ifnot = lib.ifnot

def redmul(*args):
    "reduces args with multiplication"
    return reduce(op.mul, args)

def locations(shape):
    return itt.product(*map(xrange, shape))


def contiguous_regions(binarr):
    """    
    Given a binary 2d array, returns a sorted (by size) list of contiguous
    regions (True everywhere)
    Version without recursion. Relies on scipy.ndimage
    """    
    sh = binarr.shape
    regions = [[]]
    visited = np.zeros(sh, bool)
    N = redmul(*sh)

    regions = []
    labels, nlab = ndimage.label(binarr)
    for j, o in enumerate(ndimage.find_objects(labels)):
        #sys.stderr.write('\rlocation %06d out of %06d'%(j+1, nlab))
        origin =  np.asarray([x.start for x in o])
        x1 = np.asarray(np.where(labels[o] == j+1)).T
        regions.append( map(tuple, x1 + origin))
    
    regions.sort(key = lambda x: len(x), reverse=True)
    return map(lambda x: RegionND(x, binarr.shape), regions)


def neighbours_x(loc,shape):
    n = len(loc)
    d = np.diag(np.ones(n))
    x = np.concatenate((d,-d)) + loc
    return filter(partial(valid_loc, shape=shape), map(tuple, x))
    

def neighbours_2(loc, shape):
    "list of adjacent locations"
    r,c = loc
    return filter(lambda x: valid_loc(x, shape),
                  [(r,c+1),(r,c-1),(r+1,c),(r-1,c),
                   (r-1,c-1), (r+1,c-1), (r-1, c+1), (r+1,c+1)])

neighbours = neighbours_x

def valid_loc(loc,shape):
    "location not outside bounds"
    return reduce(op.__and__, [(0 <= x < s) for x,s in zip(loc,shape)])


def filter_proximity(mask, rad=3, size=5, fn = lambda m,i,j: m[i,j]):
    rows, cols = mask.shape
    X,Y = np.meshgrid(xrange(cols), xrange(rows))
    in_circle = lib.in_circle
    out = np.zeros((rows,cols), np.bool)
    for row in xrange(rows):
        for col in xrange(cols):
            if fn(mask,row,col):
                a = in_circle((col,row),rad)
                if np.sum(mask*a(X,Y))>size:
                    out[row,col] = True
    return out

def majority(mask, th = 5, mod = True):
    rows, cols = mask.shape
    out = np.zeros((rows,cols), np.bool)
    for row in xrange(rows):
        for col in xrange(cols):
            x = np.sum([mask[n] for n in neighbours((row,col),mask.shape)])
            out[(row,col)] = (x >= th)
            if mod:
               out[(row,col)] *= mask[row,col]
    return out
            

def filter_mask(mask, fn, args=()):
    """Split a mask into contiguous regions, filter their size,
    and return result as a mask
    """
    regs = contiguous_regions_2d(mask)
    filtered_regs = fn(regs, *args)
    z = np.zeros(mask.shape, dtype=np.bool)
    if len(filtered_regs) >1:
        return reduce(lambda a,b:a+b,
                      [z]+[r.tomask() for r in filtered_regs])
    else:
        return z

def filter_size_regions(regions, min_size=5):
    "Filters clusters by their size"
    return [r for r in regions if r.size()>min_size]

def filter_shape_regions(regions, th = 2):
    "Filters continuous regions by their shape"
    return [r for r in regions
            if (r.linsize() > th*np.sqrt(r.size()))]

def glue_adjacent_regions(regions, max_distance=10):
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
        x = filter(None, map(lambda x: _glue_if(first,x),rest))
        if x == []:
            acc.append(first)
            _loop(rest)
        else:
            a = reduce(_glue_if, x)
            _loop([a] + [b for b in rest if not regions_overlap(a,b)])

    _loop(regions)
    return acc

def regions_overlap(r1,r2):
    x = False
    for loc in r1.locs:
        if loc in r2.locs:
            return True
    return False
    #return x #or y
        
def unite_2regions(region1,region2):
    "Glue together two regions"
    return RegionND(list(region1.locs) + list(region2.locs), region1.shape)
    return


def distance_regions(r1, r2, fn=min, start=1e9):
    dists = [lib.eu_dist(*pair) for pair in
             itt.product(r1.borders(), r2.borders())]
    #print dists
    return reduce(fn, dists, start)


def distance_regions_centra(r1,r2):
    return lib.eu_dist(r1.center(), r2.center())

class RegionND:
    "Basic class for a contiguous region. Can make masks from it"
    def __init__(self, locs, shape):
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
                len(filter(lambda x: x not in self.locs, neighbours(l,self.shape))))
    def linsize(self,):
        dists = [lib.eu_dist(*pair) for pair in lib.allpairs0(self.borders())]
        return reduce(max, dists, 0)
                               
        pass
    def tomask(self):
        m = np.zeros(self.shape, bool)
        for loc in self.locs: m[loc]=True
        return m


#----------------------------


