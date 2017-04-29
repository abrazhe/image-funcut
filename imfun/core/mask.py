import numpy as np
from .coords import locations

def mask_percent_threshold(mat, thresh):
    minv = np.min(mat)
    maxv = np.max(mat)
    val = (maxv-minv) * thresh/100.0
    return mat > val

def mask_threshold(mat, thresh, func=lambda a,b: a>b):
    return mat > thresh

def mask_num_std(mat, n, func=lambda a,b: a>b):
    "Same as threshold, but threshold value is times S.D. of the matrix"
    x = np.std(mat)
    return func(mat, x*n)

def mask_median_SD(mat, n = 1.5, compfn = np.greater):
    return compfn(mat, np.median(mat) + n*mat.std())

def mask_low_percentile(mat, threshold = 15.0):
    low = np.percentile(np.ravel(mat), threshold)
    return mat < low

def invert_mask(m):
    def _neg(a):
        return not a
    return np.vectorize(_neg)(m)

def zero_in_mask(mat, mask):
        out = np.copy(mat)
        out[mask] = 0.0
        return out

def zero_low_sd(mat, n = 1.5):
    return zero_in_mask(mat, mask_median_SD(mat,n,np.less))


def mask2points(mask):
    "mask to a list of points, as row,col"
    points = []
    for loc in locations(mask.shape):
        if mask[loc]:
            points.append(loc) 
    return points

    
def mask2pointsr(mask):
    "mask to a list of points, with X,Y coordinates reversed"
    points = []
    for loc in locations(mask.shape):
        if mask[loc]:
            points.append(loc[::-1]) 
    return points

def array2points(arr):
    return [r for r in surfconvert(arr)]
    
def surfconvert(frame, mask):
    from .array_handling import rescale
    out = []
    nr,nc = list(map(float, frame.shape))
    space_scale = max(nr, nc)
    f = rescale(frame)
    for r in range(int(nr)):
        for c in range(int(nc)):
            if not mask[r,c]:
                out.append([c/space_scale,r/space_scale, f[r,c]])
    return np.array(out)
