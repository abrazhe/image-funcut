""" Functions for Multiscale vision model implementation """

import itertools as itt

import numpy as np
from scipy import ndimage

from imfun import lib
from imfun import atrous
from imfun import cluster

import sys

_dtype_ = atrous._dtype_
_show_time_ = False

def with_time_dec(fn):
    "take a function and timer its evaluation"
    def _(*args, **kwargs):
	import time
	t = time.time()
	out = fn(*args,**kwargs)
        if _show_time_:
            print "time lapsed %03.3e in %s"%(time.time() - t, str(fn))
	return out
    return _


class MVMNode:
    def __init__(self, ind, labels, xslice, max_pos, level, stem=None):
        self.branches = []
        self.stem = stem
        self.ind = ind
        self.labels = labels
        self.max_pos = max_pos
        self.level = level
        self.slice = xslice
        self.mass = self.selfmass()
    def locations(self):
        return map(tuple, np.argwhere(self.labels==self.ind))
    def selfmass(self):
        return np.sum(self.labels[self.slice] == self.ind)
    def cut(self):
        return cut_branch(self.stem, self)

def engraft(stem, branch):
    "Add branch to a parent stem"
    if branch not in stem.branches:
        stem.branches.append(branch)
    branch.stem = stem
    return stem

def cut_branch(stem, branch):
    "Cut branch from a parent stem"
    if branch in stem.branches:
	stem.branches = filter(lambda x: x is not branch, stem.branches)
        branch.stem = None
    return branch

def max_pos(arr, labels, ind, xslice):
    "Position of the maximum value"
    offset = np.array([x.start for x in xslice], _dtype_)
    x = ndimage.maximum_position(arr[xslice], labels[xslice], ind)
    return tuple(x + offset)

@with_time_dec
def get_structures(support, coefs):
    acc = []
    for level,c,s in zip(xrange(len(support[:-1])), coefs[:-1], support[:-1]):
        labels,nlabs = ndimage.label(s)
        slices = ndimage.find_objects(labels)
        mps = [max_pos(c, labels, k+1, slices[k]) for k in range(nlabs)]
        acc.append([[MVMNode(k+1, labels, slices[k], mp, level)
		for k,mp in enumerate(mps)], labels])
    return acc



### NOTE: it only preserves structures that are connected up to the latest
### level this is probably not always the desired behaviour
### update: fixed, not tried yet
@with_time_dec
def connectivity_graph(structures, min_nscales=2):
    labels = [s[1] for s in structures]
    structs= [s[0]  for s in structures]
    acc = []
    for j,sl in enumerate(structs[:-1]):
        for n in sl:
            ind = labels[j+1][n.max_pos]
            if ind:
                for np in structs[j+1]:
                    if np.ind == ind:
                        engraft(np, n)
                        continue
            else: 
                acc.append(n)
    for n in structs[-1]:
	acc.append(n)
    return [s  for s in acc
            if len(s.branches) and nscales(s)>=min_nscales]


def flat_tree(root_node):
    acc = [root_node]
    for node in root_node.branches:
        acc.append(flat_tree(node))
    return lib.flatten(acc)

def nleaves(root):
    return len(flat_tree(root))

def nscales(object):
    "how many scales (levels) are linked with this object"
    levels = [n.level for n in flat_tree(object)]
    return np.max(levels) - np.min(levels) + 1

def tree_mass(root):
    return np.sum([node.mass for node in flat_tree(root)])

def tree_locations(root):
    acc = {}
    for node in flat_tree(root):
        for l in node.locations():
            if not acc.has_key(l):
                acc[l] = True
    return len(acc.keys())


def tree_locations2(root):
    out = np.zeros(root.labels.shape, np.bool)
    for node in flat_tree(root):
        out[node.labels==node.ind] = True
    return np.argwhere(out)


def all_max_pos(structures,  shape):
    out = np.zeros(shape, np.bool)
    for s in structures:
        out[s.max_pos] = True
    return out


def restore_from_locs_only(arr, object):
    locs = tree_locations2(object)
    out = np.zeros(object.labels.shape, _dtype_)
    for l in map(tuple, locs):
        out[l] = arr[l]
    return out

def restore_object(object, coefs, min_level=0):
    supp = supp_from_obj(object,min_level)
    return atrous.rec_with_support(coefs, supp)


@with_time_dec            
def supp_from_obj(object, min_level=0, verbose=0, mode=0):
    sh = object.labels.shape
    new_shape = [object.level+1] + list(sh)
    out = lib.memsafe_arr(new_shape, np.bool)
    flat = flat_tree(object)
    nfl = len(flat)
    for j,n in enumerate(flat):
	if verbose :
	    sys.stderr.write('\rnode %d out of %d'%(j+1, nfl))
        if n.level > min_level:
	    if mode == 1:
		out[n.level][n.labels==n.ind] = True
	    else:
		out[n.level][n.slice] = np.where(n.labels[n.slice]==n.ind, True, False)
	    
    return out



def deblend_node_old(node, coefs, acc = None):
    distance = cluster.euclidean
    if acc is None: acc = [node]
    flat_leaves = flat_tree(node)
    mxcoef = lambda level, loc : coefs[level][loc]
    sublevel = lambda level: [n for n in flat_leaves if (level-n.level)==1]
    if len(node.branches) > 1:
        tocut = []
        for b in node.branches:
            wjm = mxcoef(b.level, tuple(b.max_pos))
            branches  = sublevel(b.level)
            if len(branches) != 0:
                positions = [x.max_pos for x in branches]
                i = np.argmin([distance(b.max_pos, p) for p in positions])
                wjm1m = mxcoef(b.level-1, tuple(positions[i]))
            else:
                wjm1m = 0
            wjp1m = np.max(coefs[b.level+1][b.labels==b.ind])
            if wjm1m < wjm > wjp1m :
                tocut.append(b)
        for c in tocut:
            acc.append(c.cut())
            deblend_node(c, coefs, acc)
    return acc

def deblend_node(node, coefs, acc = None, min_scales=2):
    distance = cluster.euclidean
    if acc is None: acc = [node]
    flat_leaves = flat_tree(node)
    mxcoef = lambda level, loc : coefs[level][loc]
    sublevel = lambda level: [n for n in flat_leaves if (level-n.level)==1]
    tocut = []
    for b in node.branches:
	wjm = mxcoef(b.level, b.max_pos)
	if len(b.branches) == 0:
	    wjm1m = 0
	else:
	    positions = [x.max_pos for x in b.branches]
	    i = np.argmin([distance(b.max_pos, p) for p in positions])
	    wjm1m = mxcoef(b.level-1, tuple(positions[i]))
	wjp1m = np.max(coefs[b.level+1][b.labels==b.ind])
	if wjm1m < wjm > wjp1m :
	    tocut.append(b)
    for c in tocut:
	free = c.cut()
	if  nscales(free) >= min_scales:
	    acc.append(free)
	    deblend_node(c, coefs, acc)
    return acc




def deblend_all(objects, coefs, min_scales=2):
    roots = lib.flatten([deblend_node(o,coefs) for o in objects])
    return [r for r in roots if nscales(r) >= min_scales]


### Below, final (recovered) objects will be represented as a pair, first
### element of which is the data sub-array, containing the object, and the
### second is a list, contaning shape and slice of the enclosing original data
### array 

@with_time_dec
def embedding(arr, delarr=True):
    sh = arr.shape
    b = np.argwhere(arr)
    starts, stops = b.min(0), b.max(0)+1
    slices = [slice(*p) for p in zip(starts, stops)]
    out =  arr[slices].copy()
    if delarr: del arr
    return out, (sh, slices)

def find_objects(arr, k = 3, level = 4, noise_std=None,
                 coefs = None,
                 supp = None,
		 start_scale = 0,
                 min_px_size = 200,
                 min_nscales = 3):
    if np.iterable(k):
        level = len(k)
    if coefs is None:
        coefs = atrous.decompose(arr, level)
    if noise_std is None:
        noise_std = atrous.estimate_sigma_mad(coefs[0])
    ## calculate support taking only positive coefficients (light sources)
    if supp is None:
        supp = atrous.get_support(coefs, np.array(k,_dtype_)*noise_std,
                                  modulus=False)  
    structures = get_structures(supp, coefs)
    g = connectivity_graph(structures)

    gdeblended = deblend_all(g, coefs, min_nscales) # destructive
    check = lambda x: len(tree_locations2(x)) > min_px_size
    objects = sorted([x for x in gdeblended if check(x)],
		     key = lambda u: tree_mass(u), reverse=True)
    pipeline = lib.flcompose(supp_from_obj,
			     lambda x:atrous.rec_with_support(coefs, x),
			     embedding)
    recovered = (pipeline(obj, start_scale) for obj in objects)
    return filter(lambda x: np.sum(x[0]>0) > min_px_size, recovered)
# ----------


def embedded_to_full(x):
    """Restores 'full' object from it's embedding, 
    e.g. full image from  object subframe
    """
    data, (shape, xslice) = x
    out = np.zeros(shape, _dtype_)
    out[xslice] = data
    return out

def density(obj):
    data, (shape, xslice) = obj
    N = np.float(np.product(data.shape))
    return npixels(obj)/N

def npixels(obj):
    data, _ = obj
    return np.sum(data != 0)

def energy(obj):
    data, _ = obj
    return np.sum(data)

def framewise_objs(frames,*args,**kwargs):
    return 

### Obvious todo: parallelize!
@lib.with_time_dec
def framewise_find_objects(frames, min_frames=5,
			   *args, **kwargs):
    framewise = lib.with_time(list, (find_objects(f, *args, **kwargs) for f in frames))
    print "all frames computed"
    fullrestored = objects_to_array(framewise)
    if fullrestored is None:
	print "No objects in any frame"
	return
    s = ndimage.generate_binary_structure(3,2)
    labels,nlab = ndimage.label(fullrestored) # re-label 3D-contiguous objects
    "re-labeled"
    pre_objs = ndimage.find_objects(labels)
    taking = [o[0].stop-o[0].start >= min_frames for o in pre_objs]
    masks = (labels == ind+1 for ind in range(len(pre_objs))
	     if taking[ind])
    print "calculating final objects"
    objects = [embedding(np.where(m,fullrestored,0)) for m in masks]
    return objects
    

def __connect_framewise_objs(objlist):
    for frame in objlist:
	for obj in frame:
	    full = embedded_to_full(obj)

def objects_to_array(objlist):
    "used in frame-by-frame analysis"
    out = []
    sh = None
    for frame in objlist:
	if len(frame):
	    sh = frame[0][1][0]
	    break
    if sh is None:
	return None
    for frame in objlist:
	if len(frame):
	    out.append(np.sum(map(embedded_to_full, frame), axis=0))
	else:
	    out.append(np.zeros(sh, _dtype_))
    return np.array(out, _dtype_)

# scratchpad


import pickle
def test(arr, prefix='seq-'):
    objects = list(find_objects(arr, k = 4,  min_px_size=4000))
    pickle.dump(objects, file(prefix+'-objects.pickle','w'))
    del objects

def test2(a, prefix='seq-framewise', min_frames=10):
    xx = framewise_objs(a, k=4, min_px_size=100)
    y = objects_to_array(xx)
    if y == []:
	return 
    s = ndimage.generate_binary_structure(3,2)
    labels,nlab = ndimage.label(y)
    yobjs = ndimage.find_objects(labels)
    taking = map(lambda x: x[0].stop-x[0].start >= min_frames, yobjs)
    masks = (labels == ind for ind in range(1,len(yobjs)+1) if taking[ind-1])
    objects = [embedding(np.where(m,y,0)) for m in masks]
    #pickle.dump(objects, file(prefix+'-framewise-objects.pickle','w'))
    return objects

    

