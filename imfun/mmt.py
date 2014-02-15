### Functions for Multiscale median transform and variants

from scipy.ndimage import median_filter

def decompose2d(arr2d, level,s=0):
    approx = median_filter(arr2d, 2*s+1)
    w = arrd - appr # median coefficients
    if level == 1:
        return [w, approx]
    else:
        return [w] + decompose2d(approx, level-1, 2*s)
    
