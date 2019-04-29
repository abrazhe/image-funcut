from scipy import ndimage
from .dctsplines import l2spline,l1spline
import numpy as np

def grayscale_msr(img, sigmas=(15,80,250),smoother=l2spline):
    out = np.zeros(img.shape,np.float32)
    img = img.astype(np.float32)
    for s in sigmas:
        #smoothed =
        out += np.log(img/(smoother(img,s)+0.001))
    return out/len(sigmas)
