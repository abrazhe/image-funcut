"""
Implementation of the "Converging squares" algorithm
"""

import numpy as np

## a square is a list of slices

def _child_slices(sl, step=1):
    "split a slice into two overlapping smaller slices"
    return [slice(sl.start+step,sl.stop),
            slice(sl.start,sl.stop-step)]
    
def _child_squares(square, step=1):
    "split a nD square into a list of overlapping smaller squares"
    lcsl = map(lambda s:_child_slices(s,step), square)
    return list(itt.product(*lcsl))

def square_size(sq):
    ## as it is assumed a square, only the
    ##  size in first dimension is needed
    return sq[0].stop-sq[0].start

def converge_square(m, square, step=1,
                   efunc=np.sum, min_size=1):
    """
    Converge one starting square to a small square

    Parameters:
      - `m`: input ND matrix
      - `square`: a starting structuring element, a list of slices
      - `step` : deflating coefficient
      - `efunc`: a measure function to apply to elements within a square
      - `min_size`: smallest size of the square when the algorithm is stopped

    Returns:
      - final (smallest) square which maximizes the efunc over the elements
    """
    if square_size(square) > min_size:
        chsq = _child_squares(square, step)
        x = [efunc(m[sq]) for sq in chsq]
        return converge_square(m, chsq[np.argmax(x)],
			       step, efunc, min_size)
    else:
	return square # undeflatable

def csq_find_rois(m, threshold = None,
                  stride=5,
                  reduct_step=1, efunc=np.mean,
                  min_size = 1):
    """
    Find regions of interest in an image with converging squares algorithm

    Parameters:
      - `m`: an N-dimensional matrix
      - `threshold`: a threshold whether a local starting square should be
                     taken into account
      - `stride`: size of a starting square
      - `reduct_step`: square is reduced by this step at each iteration
      - `efunc`: a measure function to apply to elements within a square
                 [``np.sum``]
      - `min_size`: smallest size of the square when the algorithm is stopped

    Returns:
      - a list of found ROIs as minimal squares
    
    """
    if threshold is None:
        threshold = np.std(m)
    cs = lambda s: converge_square(m,s,reduct_step,efunc,min_size)
    rois = []
    for square in make_grid(m.shape, stride,stride):
        if efunc(m[square]) > threshold:
            rois.append(cs(square))
    return rois

def csq_plot_rois(m,rois):
    """
    A helper function to plot the ROIs determined by the
    csq_find_rois implementation of the converging squares algorithm

    Parameters:
      - `m` : a 2D matrix
      - `rois`: a list of ROIs
    """
    import pylab as pl
    pl.figure()
    pl.imshow(m, aspect='equal', cmap='gray')
    positions = [[[s.start] for s in r[::-1]] for r in rois]
    points = map(csqroi2point, rois)
    for p in points:
        pl.plot(*p,ls='none',color='r',marker='s')

def csqroi2point(roi):
    """Helper function, converts a ROI to a point to plot"""
    return [s.start for s in roi[::-1]] # axes are reverse to indices
    

def make_grid(shape,size,stride):
    """Make a generator over sets of slices which go through the provided shape
       by a stride
    """
    origins =  itt.product(*[range(0,dim,stride) for dim in shape])
    squares = ([slice(a,a+size) for a in o] for o in origins)
    return squares
