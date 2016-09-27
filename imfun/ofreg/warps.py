import numpy as np

from scipy.ndimage.interpolation import map_coordinates

_boundary_mode = 'constant'

def flow_from_fn(fn, sh):
    start_coordinates = np.meshgrid(*map(np.arange, sh[::-1]))
    return (fn(start_coordinates)-np.array(start_coordinates))

def apply_warp(img, warp ,mode=_boundary_mode):
    """Given an image and a function to warp coordinates,
    or a pair (u,v) of horizontal and vertical flows
    warp image to the new coordinates.
    In case of a multicolor image, run this function for each color"""
    sh = img.shape
    if np.ndim(img) == 2:
        start_coordinates = np.meshgrid(*map(np.arange, sh[:2][::-1]))
        if callable(warp):
            new_coordinates = warp(start_coordinates)
        elif isinstance(warp, (tuple, list, np.ndarray)):
            new_coordinates = [c+f for c,f in zip(start_coordinates, warp)]
        else:
            raise ValueError("warp can be either a function or an array")
        return map_coordinates(img, new_coordinates[::-1], mode=mode)
    elif np.ndim(img) > 2:
        return np.dstack([apply_fn_warp(img[...,c],fn,mode) for c in range(img.shape[-1])])
    else:
        raise ValueError("Can't handle image of such shape: {}".format(sh))
