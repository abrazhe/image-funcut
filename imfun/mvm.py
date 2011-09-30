""" Functions for Multiscale vision model implementation """

import itertools as itt

import numpy as np
from scipy import ndimage

from imfun import lib
from imfun import atrous
from imfun import cluster


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
        stem.branches = [b for b in stem.branches if b is not branch]
        branch.stem = None
    return branch

def max_pos(arr, labels, ind, xslice):
    "Position of the maximum value"
    offset = np.array([x.start for x in xslice])
    x = ndimage.maximum_position(arr[xslice], labels[xslice], ind)
    return tuple(x + offset)

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
def connectivity_graph(structures, min_nscales=3):
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
    out = np.zeros(shape)
    for s in structures:
        out[s.max_pos] = True
    return out


def restore_from_locs_only(arr, object):
    locs = tree_locations2(object)
    out = np.zeros(object.labels.shape)
    for l in map(tuple, locs):
        out[l] = arr[l]
    return out

def restore_object(object, coefs, min_level=0):
    supp = supp_from_obj(object,min_level)
    return atrous.rec_with_support(coefs, supp)

def nscales(object):
    "how many scales (levels) are linked with this object"
    levels = [n.level for n in flat_tree(object)]
    return np.max(levels) - np.min(levels) + 1

            
def supp_from_obj(object, min_level=1):
    sh = object.labels.shape
    new_shape = [object.level+1] + list(sh)
    out = np.zeros((new_shape), np.bool)
    for n in flat_tree(object):
        if n.level > min_level:
            out[n.level][n.labels==n.ind] = True
    return out



def deblend_node(node, coefs, acc = None):
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

                    

def deblend_all(objects, coefs):
    return lib.flatten([deblend_node(o,coefs) for o in objects])


def embedding(arr):
    sh = arr.shape
    b = np.argwhere(arr)
    starts, stops = b.min(0), b.max(0)+1
    slices = [slice(*p) for p in zip(starts, stops)]
    return arr[slices], (sh, slices)

def find_objects(arr, k = 3, level = 4, noise_std=None,
                 min_px_size = 200,
                 min_nscales = 3):
    if np.iterable(k):
        level = len(k)
    coefs = atrous.decompose(arr, level)
    if noise_std is None:
        noise_std = atrous.estimate_sigma_mad(coefs[0])
    supp = atrous.get_support(coefs, np.array(k)*noise_std)
    structures = get_structures(supp, coefs)
    g = connectivity_graph(structures)

    gdeblended = deblend_all(g, coefs) # destructive
    check = lambda x: (nscales(x) >= min_nscales) and (len(tree_locations2(x)) > min_px_size)
    x = sorted([x for x in gdeblended if check(x)],
               key = lambda u: tree_mass(u), reverse=True)
    waves = (atrous.rec_with_support(coefs, supp_from_obj(o,0)) for o in x)
    waves = itt.imap(embedding, waves)
    for w,bounds in waves:
        if np.any(w>0) and np.sum(w>0) > min_px_size:
            yield w*(w>0), bounds


# ----------

th = 3*0.7
#coefs = lib.with_time(dec_semisep_atrous, sn.data, 4)

def embedded_to_full(x):
    data, (shape, xslice) = x
    out = np.zeros(shape)
    out[xslice] = data
    return out

def framewise_objs(frames,*args,**kwargs):
    return [list(find_objects(f, *args, **kwargs)) for f in frames]

def connect_framewise_objs(objlist):
    for frame in objlist:
	for obj in frame:
	    full = embedded_to_full(obj)

def objects_to_array(objlist):
    out = []
    for frame in objlist:
	out.append(np.sum(map(embedded_to_full, frame), axis=0))
    return np.array(out)

# scratchpad


import pickle
def test(a):
    objects1 = find_objects(a, k = 4,  min_px_size=5000)
    for j,o in enumerate(objects1):
	pickle.dump(o, file('seq-3-wave-%d.pickle'%j,'w'))
        #np.save('seq-3-wave-%d'%j, o)
        del o
        print j

def test2(a):
    xx = framewise_objs(s1n.data, k=4, min_nscales=3)
    print "1"
    y = objects_to_array(xx)
    print '2'
    labels,nlab = ndimage.label(y)
    yobjs = ndimage.find_objects(labels)
    print '3'
    taking = map(lambda x: x[0].stop-x[0].start > 1, yobjs)
    print '4'
    masks = (labels == ind for ind in range(1,len(yobjs)+1) if taking[ind-1])
    for j,m in enumerate(masks):
	print j
	o = where(m,y,0)
	np.save('seq-3-wave-framewise-%d'%j,o)
	del o
    

