## some routines for filtering
from __future__ import division
import numpy as np
from scipy import signal
from scipy import ndimage
from scipy.ndimage import convolve1d

import itertools as itt
import bwmorph


def locations(shape):
    return itt.product(*map(xrange, shape))


def gauss_kern(xsize=1.5, ysize=None):
    """ Returns a normalized 2D gauss kernel for convolutions """
    xsize = int(xsize)
    ysize = ysize and int(ysize) or xsize
    x, y = np.mgrid[-xsize:xsize+1, -ysize:ysize+1]
    g = np.exp(-(x**2/float(xsize) + y**2/float(ysize)))
    return g / g.sum()


def gauss_blur(X,size=1.0):
    return signal.convolve2d(X,gauss_kern(size),'same')

def in_range(low, high):
    return lambda x: (x >=low)*(x < high)

## this is used for noise estimation and support calculation
sigmaej = [[0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],   # 0D 
           [0.700, 0.323, 0.210, 0.141, 0.099, 0.071, 0.054],   # 1D
           [0.889, 0.200, 0.086, 0.041, 0.020, 0.010, 0.005],   # 2D
           [0.956, 0.120, 0.035, 0.012, 0.004, 0.001, 0.0005]] # 3D

# TODO: make it faster
def adaptive_medianf(arr, k = 2):
    sh = arr.shape
    out = arr.copy()
    for row in xrange(1,sh[0]-1):
        for col in xrange(1,sh[1]-1):
            sl = (slice(row-1,row+1), slice(col-1,col+1))
            m = np.mean(arr[sl])
            sd = np.std(arr[sl])
            if (arr[row,col] > m+k*sd) or \
                   (arr[row,col] < m- k*sd):
                out[row, col] = np.median(arr[sl])
    return out
    


def opening_of_closing(a):
    "performs binary opening of binary closing of an array"
    bclose = ndimage.binary_closing
    bopen = ndimage.binary_opening
    return bopen(bclose(a))


def dec_atrous2d(arr2d, lev, kern=None, boundary='symm'):
    """
    Do 2d a'trous wavelet transform with B3-spline scaling function

    This is a convolution version, where kernel is zero-upsampled
    explicitly. Not fast.

    Inputs:
    ---------
    arr2d : 2D array
    kern  : low-pass filter kernel (B3-spline by default)
    boundary : boundary conditions (passed to scipy.signal.convolve2d, 'symm'
               by default)
    Outputs:
    ---------
    list of wavelet details + last approximation
    
    """
    _b3spline1d = np.array([1./16, 1./4, 3./8, 1./4, 1./16])
    __x = _b3spline1d.reshape(1,-1)
    _b3spl2d = np.dot(__x.T,__x)
    if kern is None: kern = _b3spl2d
    if lev <= 0: return arr2d
    shapecheck = map(lambda a,b:a>b, arr2d.shape, kern.shape)
    assert np.all(shapecheck)
    # approximation:
    approx = signal.convolve2d(arr2d, kern, mode='same',
                               boundary=boundary)  
    w = arr2d - approx   # wavelet details
    upkern = zupsample(kern)
    shapecheck = map(lambda a,b:a>b, arr2d.shape, upkern.shape)
    if lev == 1:
        return [w, approx]
    elif not np.all(shapecheck):
        print "Maximum possible decomposition level reached, not advancing any more"
        return [w, approx]
    else:
        return [w] + dec_atrous2d(approx,lev-1,upkern,boundary) 

def f2d(phi):
    v = phi.reshape(1,-1)
    return np.dot(v.T,v)

def dec_semisep_atrous(arr, level=1,
                         phi = np.array([1./16, 1./4, 3./8, 1./4, 1./16])):
    "Semi-separable atrous ..."
    from swan import pydwt
    phi2d = f2d(phi)
    if level <= 0: return arr
    tapprox = np.zeros(arr.shape)
    for loc in locations(arr.shape[1:]):
        v = arr[:,loc[0], loc[1]]
        tapprox[:,loc[0], loc[1]] = convolve1d(v, phi, mode='mirror')
    approx = np.zeros(arr.shape)
    for k in xrange(arr.shape[0]):
        approx[k] = signal.convolve2d(tapprox[k], phi2d, mode='same', boundary='symm')
    details = arr - approx
    upkern = zupsample(phi)
    shapecheck = map(lambda a,b:a>b, arr.shape, upkern.shape)
    if level == 1:
        return [details, approx]
    elif not np.all(shapecheck):
        print "Maximum allowable decomposition level reached, returning"
        return [details, approx]
    else:
        return [details] + dec_semisep_atrous(approx, level-1, upkern)

def zupsample(arr):
    "Upsample array by interleaving it with zero values"
    sh = arr.shape
    newsh = [d*2-1 for d in sh]
    o = np.zeros(newsh,dtype=arr.dtype)
    o[[slice(None,None,2) for d in sh]] = arr
    return o


def rec_atrous(coefs, level=None):
    "Reconstruct from a trous decomposition. Last coef is last approx"
    return np.sum(coefs[-1:level:-1], axis=0)

def represent_support(supp):
    out = [2**(j+1)*supp[j] for j in range(len(supp)-1)]
    return np.sum(out, axis=0)

def get_support(coefs, th, neg=False):
    out = []
    nd = len(coefs[0].shape)
    fn = neg and np.less or np.greater
    for j,w in enumerate(coefs[:-1]):
        t  = np.iterable(th) and th[j] or th
        out.append(fn(np.abs(w), t*sigmaej[nd][j]))
    out.append(np.ones(coefs[-1].shape)*(not neg))
    return out

def invert_mask(m):
    def _neg(a):
        return not a
    return np.vectorize(_neg)(m)

def arr_or(a1,a2):
    return np.vectorize(lambda x,y: x or y)(a1,a2)

def estimate_sigma(arr, coefs, k=3, eps=0.01, max_iter=1e9):
    sprev = estimate_sigma_mad(coefs[0])
    #sprev = arr.std()
    for j in xrange(int(max_iter)):
        supp = get_support(coefs, sprev*k, neg=True)
        mask = np.prod(supp[:-1], axis=0)
        snext =  np.std((arr-coefs[-1])[mask])
        #print snext, sprev
        assert np.sum(mask) > 0
        if abs(sprev-snext)/snext <= eps:
            return snext
        sprev = snext
    return sprev


def estimate_sigma_mad(coefarr):
    return np.median(np.abs(coefarr))/(0.6745*sigmaej[2][0])

def wavelet_enh_std(f, level=4, out = 'rec',absp = False):
    fw = dec_atrous2d(f, level)
    if absp:
        supp = map(lambda x: abs(x) > x.std(), fw)
    else:
        supp = map(lambda x: x > x.std(), fw)
    if out == 'rec':
        filtcoef = [x*w for x,w in zip(supp, fw)]
        return rec_atrous(filtcoef)
    elif out == 'supp':
        return represent_support(supp)

def rec_with_supp(coefs, supp):
    return rec_atrous([c*s for c,s in zip(coefs, supp)])

        
def wavelet_denoise(f, k=[3,3,2,2], level = 4, noise_std = None):
    dim = len(f.shape)
    if dim == 2:
        decfn = dec_atrous2d
    elif dim == 3:
        decfn = dec_semisep_atrous
    else:
        print "Data dimension not supported"
        return
    if np.iterable(k):
        level = len(k)
    coefs = decfn(f, level)
    if noise_std is None:
        if dim < 3:
            noise_std = estimate_sigma(f, coefs) / 0.974 # magic value
        else:
            noise_std = estimate_sigma_mad(coefs[0])
    supp = get_support(coefs, np.array(k)*noise_std)
    filtcoef =  [c*s for c,s in zip(coefs, supp)]
    return rec_atrous(filtcoef)

### ---- MVM ---------

#from imfun import bwmorph

def engraft(stem, branch):
    if branch not in stem.branches:
        stem.branches.append(branch)
    branch.stem = stem
    return stem

def cut_branch(stem, branch):
    if branch in stem.branches:
        stem.branches = [b for b in stem.branches if b is not branch]
        branch.stem = None
    return branch

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


def get_structures(support, coefs):
    for level,c,s in zip(xrange(len(support[:-1])), coefs[:-1], support[:-1]):
        labels,nlabs = ndimage.label(s)
        slices = ndimage.find_objects(labels)
        mps = [max_pos(c, labels, k+1, slices[k]) for k in range(nlabs)]
        yield [MVMNode(k+1, labels, slices[k], mp, level) for k,mp in enumerate(mps)]
                     
        
def max_pos(arr, labels, ind, xslice):
    offset = np.array([x.start for x in xslice])
    x = ndimage.maximum_position(arr[xslice], labels[xslice], ind)
    return tuple(x + offset)

def connectivity_graph(structures, min_nscales=3):
    labels = [s[0].labels for s in structures]
    for j,sl in enumerate(structures[:-1]):
        for n in sl:
            ind = labels[j+1][n.max_pos]
            if ind:
                for np in structures[j+1]:
                    if np.ind == ind:
                        engraft(np, n)
                        continue
    return [s  for s in structures[-1]
            if len(s.branches) and nscales(s)>=min_nscales]
                          

def all_max_pos(structures,  shape):
    out = np.zeros(shape)
    for s in structures:
        out[s.max_pos] = True
    return out

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

def restore_from_locs_only(arr, object):
    locs = tree_locations2(object)
    out = np.zeros(object.labels.shape)
    for l in map(tuple, locs):
        out[l] = arr[l]
    return out

def restore_object(object, coefs, min_level=0):
    supp = supp_from_object(object,min_level)
    return rec_with_supp(coefs, supp)

def nscales(object):
    "how many scales (levels) are linked with this object"
    levels = [n.level for n in flat_tree(object)]
    return np.max(levels) - np.min(levels) + 1

            
def supp_from_object(object, min_level=1):
    sh = object.labels.shape
    new_shape = [object.level+1] + list(sh)
    out = np.zeros((new_shape), np.bool)
    for n in flat_tree(object):
        if n.level > min_level:
            out[n.level][n.labels==n.ind] = True
    return out


from cluster import euclidean as distance

def deblend_node(node, coefs, acc = None):
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

                    

def deblend_all(objects,coefs):
    return lib.flatten([deblend_node(o,coefs) for o in objects])


def embedding(arr, ax=None):
    b = np.argwhere(arr)
    starts, stops = b.min(0), b.max(0)+1
    if ax is None:
        slices = [slice(*p) for p in zip(starts, stops)]
	return arr[slices]
    else:
        return arr[starts[0]:stops[0],:,:]

def find_objects(arr, k = 3, level = 4, noise_std=None,
                 strip = False,
                 min_px_size = 200,
                 min_nscales = 3):
    ndim = arr.ndim
    if ndim == 2:
        decompose = dec_atrous2d
    elif ndim == 3:
        decompose = dec_semisep_atrous
    else:
        print "Cant work in %d dimensions" %ndim
        return
    print "Calculating decomposition..."
    coefs = lib.with_time(decompose, arr, level)
    if noise_std is None:
        noise_std = estimate_sigma_mad(coefs[0])
    supp = get_support(coefs, k*noise_std)

    print 'Calculating structures'
    structures = lib.with_time(list, get_structures(supp, coefs))

    print "Calculating connectivity graph"
    g = lib.with_time(connectivity_graph, structures)
    gdeblended = deblend_all(g, coefs) # destructive
    check = lambda x: (nscales(x) >= min_nscales) and (len(tree_locations2(x)) > min_px_size)
    print "Sorting and pre-pruning deblended objects"
    x = sorted([x for x in gdeblended if check(x)],
               key = lambda u: tree_mass(u), reverse=True)
    print "starting object generator"            
    waves = (rec_with_supp(coefs, supp_from_object(o,0)) for o in x)
    if strip:
        waves = itt.imap(lambda x: embedding(x,0), waves)
    for w in waves:
        if np.any(w>0) and np.sum(w>0) > min_px_size:
            yield w*(w>0)



# ----------

th = 3*0.7
#coefs = lib.with_time(dec_semisep_atrous, sn.data, 4)


def test(a):
    objects1 = find_objects(a, k = 4, strip=False, min_px_size=5000)
    for j,o in enumerate(objects1):
        np.save('seq-3-wave-%d'%j, o)
        del o
        print j

    

