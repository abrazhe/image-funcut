"""
---  Distance measures ---
"""

import numpy as np
from scipy import stats


def minkowski(p1,p2,k):
    '''Minkowski distance between points p1 and p2
    if k equals 1, it is a Manhattan (cityblock) distance
    if k equals 2, it is a Euclidean distance
    '''
    if  isinstance(p1, np.ndarray) and isinstance(p2, np.ndarray):
        return np.power(np.sum(np.abs((p1-p2)).T**k, 0), 1./k) # do axis sum
                                        # to allow for vectorized input
    else:
        x = list(map(lambda x,y: abs((x-y)**k), p1, p2))
        return np.power(np.sum(x), 1./k)

def euclidean(p1,p2):
    "Euclidean distance between 2 points"
    return minkowski(p1,p2,2)

def cityblock(p1,p2):
    "Cityblock (Manhattan) distance between 2 points"
    return minkowski(p1,p2,1)

def pearson(v1,v2):
    "Pearson distance measure"
    return 1 - stats.pearsonr(v1,v2)[0]

def apearson(v1,v2):
    "Absolute Pearson distance measure"
    return 1 - abs(stats.pearsonr(v1,v2)[0])

def spearman(v1,v2):
    "Spearman distance measure"
    return 1 - stats.spearmanr(v1,v2)[0]

def xcorrdist(v1,v2):
    "Correlation distance measure"
    return 1/(np.correlate(v1,v2) + 1e-6)
