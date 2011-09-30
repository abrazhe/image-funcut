## some routines for filtering
from __future__ import division
import numpy as np
from scipy import signal
from scipy import ndimage

import itertools as itt


def gauss_kern(xsize=1.5, ysize=None):
    """ Returns a normalized 2D gauss kernel for convolutions """
    xsize = int(xsize)
    ysize = ysize and int(ysize) or xsize
    x, y = np.mgrid[-xsize:xsize+1, -ysize:ysize+1]
    g = np.exp(-(x**2/float(xsize) + y**2/float(ysize)))
    return g / g.sum()


def gauss_blur(X,size=1.0):
    return signal.convolve2d(X,gauss_kern(size),'same')



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





