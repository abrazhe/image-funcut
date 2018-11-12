# Register stacks/collections of images

import itertools as itt
from functools import partial

import numpy as np
from imfun import core
from imfun import fseq

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


from imfun.components import pca
from imfun import cluster

from . import warps
from .warps import Warp

from collections import defaultdict

def make_transition_map(affs):
    Nex = len(np.unique(affs))
    x = defaultdict(lambda : defaultdict(lambda :0))
    for i in range(len(affs)-1):
        if affs[i] != affs[i+1]:
            x[affs[i]][affs[i+1]] += 1
            x[affs[i+1]][affs[i]] += 1
    return {n:dict(x[n]) for n in x}

from functools import reduce
import operator as op


class TemplateNode:
    def __init__(self,idx,tmap):
        self.idx = idx
        self.tmap = tmap
        self.up = None
        self.down = []
    def add_down(self, node):
        if node != self:
            #self.down[node] = self.affinity(node)
            self.down.append(node)
            node.add_up(self)
    def affinity(self,node):
        return self.tmap[self.idx][node.idx]
    def add_up(self,node):
        self.up = node
    def unroot(self):
        up = self.up
        self.up = None
        up.down = [b for b in up.down if b != self]
    def is_upstream(self,node):
        if self.up is None: return False
        if node == self.up: return True
        return self.up.is_upstream(node)
    def __repr__(self):
        #return str(self.idx)
        #return repr([self.idx, self.up.idx if self.up else None, [b for b in self.down]])
        return repr([self.idx, [b for  b in self.down] if len(self.down) else None])
        
def tmap_to_tree(tmap,seed_index,visited=None):
    if visited is None:
        visited = {}
    #print('seed:',seed_index)
    #print('visited:', list(visited.keys()))
    if seed_index in visited:
        root = visited[seed_index][0]
        visited[seed_index][1] += 1
    else:
        root = TemplateNode(seed_index,tmap)
        visited[seed_index] = [root, 0]
    if visited[seed_index][1] > 10:
        return
    for j in tmap[seed_index]:
        #print('%d->%d'%(seed_index,j))        
        if j not in visited:
            #print('creating new node: %d'%j)
            branch = TemplateNode(j,tmap)
            root.add_down(branch)
            visited[j] = [branch,0]
        else:
            branch,_ = visited[j]          
        if (root.up is branch) or branch.up is None:
            #print('breaking branch %d'%branch.idx)
            #print('up is None' if branch.up is None else 'root.up is branch: %d'%root.up.idx)
            continue
        if root.affinity(branch) > branch.affinity(branch.up) and not root.is_upstream(branch):
            #print('-------------- re-routing branch %d from %d to %d' %(branch.idx, branch.up.idx, root.idx))
            #up = branch.up
            branch.unroot()
            root.add_down(branch)
            #root.up = up
            pass
        if not len(branch.down):
            tmap_to_tree(tmap, j, visited)
        else:
            pass
            #print('skipping visited')

    return root

def calc_paths(node, acc=None, out = None):
    if acc is None: acc = []
    if out is None: out = {}
    acc = acc + [node.idx]
    out[node.idx] = acc[::-1]
    for b in node.down:
        calc_paths(b,acc,out)
    return out

def zero_shift(shape):
    return Warp.from_array([1e-5*np.zeros(shape),1e-3*np.zeros(shape)])

def calc_paths(node, acc=None, out = None):
    if acc is None: acc = []
    if out is None: out = {}
    acc = acc + [node.idx]
    out[node.idx] = acc[::-1]
    for b in node.down:
        calc_paths(b,acc,out)
    return out

def calc_transitions_plain(templates, seed, regfn, **kwargs):
    out = {}
    for j in range(len(templates)):
        if j == seed:
            out[j] = zero_shift(templates[seed].shape)
        else:
            out[j] = regfn(templates[j],templates[seed], **kwargs)
    return out

def calc_transitions1(templates, paths, regfn, **kwargs):
    out = {}
    cache = {} # todo: use cache of transforms
    for j in paths:
        path = paths[j]
        if len(path)>1:
            ws = [regfn(templates[pfrom],templates[pto], **kwargs) for pfrom,pto in zip(path,path[1:])]
            out[j] = reduce(op.add, ws)
        else:
            out[j] = zero_shift(templates[0].shape)
    return out

def to_templates(frames, templates, index, regfn, njobs=4, **fnargs):

    # first template is the "master" template because it is based on the largest
    # number of frames. All other templates must eventually be registered to this one
    # There may be other criteria, such as the most contrast or the sharpest image
    all_warps = []

    print('templates: preparing transition map')
    tmap = make_transition_map(index)
    seed = len(np.unique(index))//2
    tgraph = tmap_to_tree(tmap, seed)
    correction_paths = calc_paths(tgraph)

    print('templates: calculating inter-template corrections')
    corrections = calc_transitions1(templates, correction_paths, regfn, **fnargs)
    
    for k,template in enumerate(templates):
        print("Aligning template %d of %d with %d members"%(k+1, len(templates), np.sum(index==k)))
        #if k > 0:
        #    correction = regfn(template, templates[0],**fnargs)
        #else:
        #    correction = Warp.from_array(np.zeros((2,)+frames[0].shape))
        correction = corrections[k]
        frame_idx = np.arange(len(frames))[index==k]
        #print len(frame_idx)
        frame_slice = (frames[fi] for fi in frame_idx)
        warps_ = to_template(frame_slice, template, regfn, njobs=njobs, **fnargs)
        #print 'Warps?', type(correction), type(warps_[0])
        #print len(warps_)
        all_warps.extend([(fi, correction+w) for fi,w in zip(frame_idx, warps_)])
        #warps = [(fi,correction+reg_fn(template,frames[fi],**fnargs)) for fi in frame_idx]
    all_warps.sort(key = lambda aw:aw[0])
    return [w[1] for w in all_warps]



def to_template(frames, template, regfn, njobs=4,  **fnargs):
    """
    Given stack of frames (or a FSeq obj) and a template image,
    align every frame to template and return a collection of functions,
    which take image coordinates and return warped coordinates, which whould align the
    image to the template.
    """
    if njobs > 1 and _with_pathos_:
        pool = ProcessPool(nodes=njobs)
        out = pool.map(partial(regfn, template=template, **fnargs), frames)
        #pool.close() ## doesn't work when this is active
    else:
        print('Running in one process')
        out = [regfn(img, template, **fnargs) for img in frames]
    return out

def to_updated_template(frames, template, regfn, update_rate=0.1, **fnargs):
    warps = []
    summed = np.copy(template)
    for i,f in enumerate(frames):
        w = regfn(f,template)
        warps.append(w)
        fc = w(f)
        summed += fc
        template = summed/(i+2)
        #template = (1-update_rate)*template + update_rate*fc
    return warps

def recursive(frames, regfn):
    """
    Given stack of frames,
    align frames recursively and return a mean frame of the aligned stack and
    a list of functions, each of which takes an image and return warped image,
    aligned to this mean frame.
    """
    #import sys
    #sys.setrecursionlimit(len(frames))
    L = len(frames)
    if L < 2:
        return frames[0], [Warp.from_function(lambda f:f)]
    else:
        mf_l, warps_left = recursive(frames[:L//2], regfn)
        mf_r, warps_right = recursive(frames[L//2:], regfn)
        fn = regfn(mf_l, mf_r)
        fm = 0.5*((mf_l,fn) + mf_r)
        return fm, [core.fnutils.flcompose(fx,fn) for fx in warps_left] + warps_right
        #return fm, [fnutils.flcompose2(fn,fx) for fx in fn1] + fn2
