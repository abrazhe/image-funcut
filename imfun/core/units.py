import numpy as np


def quantity_to_pair(q):
    return q.value, str(q.unit)

def quantity_to_scale(q):
    return pair_to_scale(quantity_to_pair(q))
    
def alist_to_scale(alist):
    names = ("scale", "units")
    formats = ('float', "S10")
    while len(alist) < 3:
        alist.append(alist[-1])
    return np.array(alist, dtype=dict(names=names,formats=formats))

def pair_to_scale(pair):
    names = ("scale", "units")
    formats = ('float', "S10")
    return np.array(pair, dtype=dict(names=names, formats=formats))















