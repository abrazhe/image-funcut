import numpy as np

from imfun.external.physics import Q

names = ("value", "unit")
formats = ('float', "U21")


def quantity_to_pair(q):
    return q.value, str(q.unit)


def pair_to_scale(pair):
    return np.array(pair,
                    dtype=dict(names=names,formats=formats)).view(np.recarray)


def unpair_to_scale(*pair):
    return pair_to_scale(pair)


QS = unpair_to_scale


def quantity_to_scale(q):
    return pair_to_scale(quantity_to_pair(q))


def quantities_to_alist(quants):
    return [quantity_to_pair(q) for q in quants]


def quantities_to_scales(quants):
    return alist_to_scale(quantities_to_alist(quants))


def alist_to_scale(alist):
    while len(alist) < 3:
        alist.append(alist[-1])
    return np.array(alist, dtype=dict(names=names,
                                      formats=formats)).view(np.recarray)


def scales_to_quantities(scales):
    return [Q(s['value'], s['unit'].astype(str)) for s in scales]
