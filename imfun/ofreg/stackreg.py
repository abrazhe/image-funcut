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


def to_templates(frames, templates, index, regfn, njobs=4, **fnargs):

    # first template is the "master" template because it is based on the largest
    # number of frames all other templates must eventually be registered to this one
    # There may be other criteria, such as the most contrast or the sharpest image
    all_warps = []

    for k,template in enumerate(templates):
        print("Doing template %d"%k)
        correction = Warp.from_array(np.zeros((2,)+frames[0].shape))
        if k > 0:
            correction = regfn(template, templates[0],**fnargs)
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
        #pool.close()
    else:
        print('Running on one core')
        out = [regfn(img, template, **fnargs) for img in frames]
    return out

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
        return frames[0], [lambda f:f]
    else:
        mf_l, warps_left = register_stack_recursive(frames[:L/2], regfn)
        mf_r, warps_right = register_stack_recursive(frames[L/2:], regfn)
        fn = regfn(mf_l, mf_r)
        fm = 0.5*(apply_fn_warp(mf_l,fn) + mf_r)
        return fm, [core.fnutils.flcompose(fx,fn) for fx in warps_left] + warps_right
        #return fm, [fnutils.flcompose2(fn,fx) for fx in fn1] + fn2
