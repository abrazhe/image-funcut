import numpy as np

def rezip(a):
    return zip(*a)

def pair_to_scale(pair):
    names = ("scale", "units")
    formats = ('float', "S10")
    return np.array(pair, dtype=dict(names=names, formats=formats))

def arr_or(a1,a2):
    return np.vectorize(lambda x,y: x or y)(a1,a2)

def ma2d(m, n):
    "Moving average in 2d (for rows)"
    for i in xrange(0,len(m)-n,):
        yield np.mean(m[i:i+n,:],0)


def __best (scoref, lst):
    if len(lst) > 0:
        n,winner = 0, lst[0]
        for i, item in enumerate(lst):
            if  scoref(item, winner): n, winner = i, item
            return n,winner
    else: return -1,None

def __min1(scoref, lst):
    return __best(lambda x,y: x < y, map(scoref, lst))


def imresize(a, nx, ny, **kw):
    """
    Resize and image or other 2D array with affine transform
    # idea from Sci-Py mailing list (by Pauli Virtanen)
    """
    from scipy import ndimage
    return ndimage.affine_transform(
        a, [(a.shape[0]-1)*1.0/nx, (a.shape[1]-1)*1.0/ny],
        output_shape=[nx,ny], **kw) 

def allpairs(seq):
    return combinations(seq,2)


def allpairs0(seq):
    if len(seq) <= 1: return []
    else:
        return [[seq[0], s] for s in seq[1:]] + allpairs(seq[1:])
