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

distance = cluster.euclidean

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

def restore_object(object, coefs, min_level=0,):
    supp = supp_from_obj(object,min_level)
    return atrous.rec_with_support(coefs, supp)


@with_time_dec            
def supp_from_obj(object, min_level=0, max_level= 10,
		  verbose=0, mode=0,
		  weights = None):
    """Support arrays from object"""
    sh = object.labels.shape
    new_shape = [object.level+1] + list(sh)
    out = lib.memsafe_arr(new_shape, _dtype_)
    flat = flat_tree(object)
    nfl = len(flat)
    if weights is None:
	weights = [1]*(object.level+1)
    for j,n in enumerate(flat):
	if verbose :
	    sys.stderr.write('\rnode %d out of %d'%(j+1, nfl))
        if n.level > min_level and n.level <=max_level:
	    val = weights[n.level]
	    if mode == 1:
		out[n.level][n.labels==n.ind] = val
	    elif mode == 2:
		out[n.level][n.labels==n.ind] = n.ind
	    else:
		out[n.level][n.slice] = np.where(n.labels[n.slice]==n.ind, val, 0)
    return out



def deblend_node_old(node, coefs, acc = None):
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
    #sublevel = lambda level: [n for n in flat_leaves if (level-n.level)==1]
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


### This is one of the main functions ###
#----------------------------------------
def find_objects(arr, k = 3, level = 5, noise_std=None,
                 coefs = None,
                 supp = None,
		 retraw = False, # only used for testing
		 start_scale = 0,
		 weights = [0.75, 1, 1, 1, 0.75],
                 min_px_size = 200,
                 min_nscales = 2):
    if np.iterable(k):
        level = len(k)
    if coefs is None:
        coefs = atrous.decompose(arr, level)
    if noise_std is None:
	if arr.ndim > 2:
	    noise_std = atrous.estimate_sigma_mad(coefs[0], True)
	else:
	    noise_std = atrous.estimate_sigma(arr, coefs)
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
    if retraw:
	return objects
    pipeline = lib.flcompose(lambda x1,x2: supp_from_obj(x1,x2,
							 weights = weights),
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

### Obvious todo: parallelize!
@lib.with_time_dec
def _framewise_find_objects_old(frames, min_frames=5,
				binary_structure = (3,2),
				*args, **kwargs):
    framewise = lib.with_time(list, (find_objects(f, *args, **kwargs) for f in frames))
    print "all frames computed"
    fullrestored = objects_to_array(framewise)
    if fullrestored is None:
	print "No objects in any frame"
	return
    s = ndimage.generate_binary_structure(*binary_structure)
    labels,nlab = ndimage.label(fullrestored) # re-label 3D-contiguous objects
    "re-labeled"
    pre_objs = ndimage.find_objects(labels)
    taking = [o[0].stop-o[0].start >= min_frames for o in pre_objs]
    masks = (labels == ind+1 for ind in range(len(pre_objs))
	     if taking[ind])
    print "calculating final objects"
    objects = [embedding(np.where(m,fullrestored,0)) for m in masks]
    return objects

def recobjs_overlap(*recobjs):
    "true if there is a non-empty intersection between reconstructed objects"
    ### note, todo: may be use a % of overlap as a threshold?
    return np.any(reduce(lambda a,b: a*b, [m>0 for m in recobjs]))

def recobjs_lm_connected(o1, o2):
    "true if position of maximum in o2 lies within o1"
    pos = ndimage.maximum_position(o2)
    return o1[pos] > 0


@lib.with_time_dec
def framewise_find_objects(frames, min_frames=5,
			   framewise=None,
			   verbose=True,
			   testfn = recobjs_overlap,
			   *args, **kwargs):
    """Framewise search for objects with multiscale vision model
    objects3D = framewise_find_objects(frames, *args, **kwargs)

    Arguments:
    ----------
    frames: list of 2D arrays or array-like.
    Output:
    ---------
    List of recovered 3D objects. Each object is a tuple, where first element
    is an embedding of the object, and second element is a list, first item of
    which is full shape of the array, and slices for the embedding.
    """
    L = len(frames)
    if framewise is None:
	framewise = []
	for fcount,frame in enumerate(frames):
	    framewise.append(find_objects(frame, *args, **kwargs))
	    if verbose:
		sys.stderr.write('\rframe %d out of %d'%(fcount+1, L))
    print ""
    linkables = make_linkable(framewise)
    res = connect_framewise_objs(linkables, testfn=testfn)
    res = [r for r in res if nframes_linked(r) >=min_frames] 
    return [restore_linked(r,L) for r in res]


def slice2range(_slice):
    "returns arange from a slice"
    return np.arange(_slice.start, _slice.stop, _slice.step)

def slices_intersect(*slices):
    "1d intersection between slices as arange"
    return reduce(np.intersect1d, map(slice2range, slices))

def xslices_intersect(*xslices):
    "n-dimensional intersection of xslices (tuples of slices)"
    return np.all([len(x) for x in map(slices_intersect, *xslices)])


class LinkableObj:
    def __init__(self, obj, framenumber):
	self.frame = framenumber
	self.obj = obj
	self.prev = []
	self.branches = []
    def isroot(self):
	return (self.prev == []) and len(self.branches)
    def linknext(self, obj):
	self.branches.append(obj)
	obj.prev.append(self)
    def linked(self, obj):
	return (obj in self.branches) or (obj in self.prev)


def make_linkable(framelist):
    return [[LinkableObj(o,j) for o in frame] for j,frame
	    in enumerate(framelist)]

def last_leaf(root):
    if root.branches == []: return root.frame
    else: return np.max([last_leaf(n) for n in root.branches])

def nframes_linked(root):
    return last_leaf(root) - root.frame + 1

def roots_only(objlist):
    return [o for o in objlist if o.isroot()]

def excluding(item, _list):
    return [e for e in _list if e is not item]

def prune_multiple_prevs(objlist):
    """
    each object in a frame n can only have one
    'ancestor' object in frame n-1"""
    for o in objlist:
	full1 = embedded_to_full(o.obj)
	test = lambda other: recobjs_lm_connected(full1, other)
	maxpos = ndimage.maximum_position(full1)
	parents = o.prev
	if len(parents) > 1:
	    fulls = [embedded_to_full(x.obj) for x in o.prev]
	    max_positions = map(ndimage.maximum_position, fulls)
	    i = np.argmin([distance(maxpos, x) for x in max_positions])
	    best_p = parents[i]
	    o.prev = [best_p]
	    for p in parents:
		if p is not best_p:
		    p.branches = excluding(o,p.branches)	    
    return objlist
	     

def connect_framewise_objs(objlist,testfn = recobjs_overlap, nnext=3):
    for j,frame in enumerate(objlist[:-nnext]):
	nextframes = objlist[j+1:j+nnext+1]
	for obj in frame:
	    full = embedded_to_full(obj.obj)
	    test1 = lambda o: testfn(full, embedded_to_full(o.obj))
	    for nf in nextframes:
		nx = filter(test1, nf)
		if len(nx):
		    for n in nx: obj.linknext(n)
		    break
    objs = [o for o in lib.flatten(objlist) if len(o.prev) or len(o.branches)]
    objs = prune_multiple_prevs(objs)
    return roots_only(objs)

def linked_to_frames(root, acc = None):
    x = (root.frame, embedded_to_full(root.obj))
    if acc is None: acc = [x]
    else: acc.append(x)
    if len(root.branches):
	for n in root.branches:
	    linked_to_frames(n, acc)
    return acc

def restore_linked(root,timespan):
    frame_shape = list(root.obj[1][0])
    first,last = root.frame, last_leaf(root)
    out = np.zeros([timespan]+frame_shape)
    for j,rec in linked_to_frames(root):
	out[j] += rec
    emb = embedding(out)
    return emb


def objects_to_array(objlist):
    "used in old version of frame-by-frame analysis"
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



#### algorithm for conjugated gradient reconstruction ####
#### -------------------------------------------------####


_phi_ = np.array([1./16, 1./4, 3./8, 1./4, 1./16], _dtype_)
__x = _phi_.reshape(1,-1)
_phi2d_ = np.dot(__x.T,__x)

c2dkw = dict(mode='same', boundary='symm')

def _atilda0(coefs):
    from imfun.fnutils import take, fniter
    kernels = take(len(coefs), fniter(atrous.zupsample, _phi2d_))
    out = np.zeros(coefs[0].shape)
    for j, c in enumerate(coefs):
	if j > 0:
	    x = atrous.signal.convolve2d(c, kernels[j-1], **c2dkw)
	else:
	    x = c
	out += x
    return out

def _atilda1(coefs):
    from imfun.fnutils import take, fniter
    kernels = take(len(coefs), fniter(atrous.zupsample, _phi2d_))
    out = np.zeros(coefs[0].shape)

    for j, c in enumerate(coefs[:-1]):
	if j > 0:
	    kern = atrous.signal.convolve2d(kernels[j-1], kern_prev, **c2dkw)
	    kern_prev = kern
	else:
	    kern = kernels[1]
	    kern_prev =kern
	x = atrous.signal.convolve2d(c, kern, **c2dkw)
	out += x
    return out

def _project(coefs, supp):
    return [c*s for c,s in zip(coefs, supp)]

def conj_grad_rec(obj, coefs, thresh = 0.1, verbose=True):
    adjoint = _atilda1
    N = len(coefs)-1
    supp = [x for x in supp_from_obj(obj)]
    while len(supp) < len(coefs):
	supp.append(np.zeros(coefs[0].shape))
    WW = _project(coefs, supp)
    F0 = atrous.rec_atrous(WW)
    x = atrous.decompose(F0, N)
    AF0 = _project(x, supp)
    Wr = [c1-c2 for c1, c2 in zip(WW, AF0)]
    Fr0 = adjoint(Wr)

    AF, Fp, Fr, niter  = AF0, F0, Fr0, 0
    conv = []
    meas_prev = -1e-5
    while niter < 1000:
	Fr_norm = np.sum(Fr**2)
	AF_norm = np.sum([x**2 for x in AF[:-1]])
	alpha = Fr_norm/AF_norm
	_Fn = Fp + alpha*Fr
	Fnext = _Fn*(_Fn >= 0)
	AF = _project(atrous.decompose(Fnext,N), supp)
	Wrn = [c1-c2 for c1, c2 in zip(WW, AF)]
	meas_next = np.sum([c**2 for c in Wrn[:-1]])
	if niter > 0:
	    meas = 1 - meas_next/meas_prev
	else:
	    meas = 1
	conv.append(meas)
	if verbose:
	    print meas, meas_next
	if meas < thresh or meas_next < thresh:
	    return Fnext, conv
	meas_prev = meas_next
	Frnext = adjoint(Wrn)
	beta = np.sum(Frnext**2)/Fr_norm
	Frnext = Frnext + beta*Fr
	Fr, Wr, Fp = Frnext, Wrn, Fnext
	niter += 1
    return Fnext
    
    
    

### 2D gaussian(s)

#### scratchpad




    

