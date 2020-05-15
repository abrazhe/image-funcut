import numpy as np
from .misc import ifnot


def DFoSD(vec, normL=None, th=1e-6):
    "Subtract mean value along first axis and normalize to S.D."
    normL = ifnot(normL, len(vec))
    m, x = np.mean, vec[:normL]
    sdx = np.std(x, 0)
    out = np.zeros(vec.shape, vec.dtype)
    if sdx.shape is ():
        if np.abs(sdx) > th:
            out = (vec - m(x)) / sdx
    else:
        zi = np.where(np.abs(sdx) < th)[0]
        sdx[zi] = -np.inf
        out = (vec - m(x)) / sdx
    return out


def DFoF(vec, normL=None, th=1e-6):
    "Subtract mean value along first axis and normalize to it"
    normL = ifnot(normL, len(vec))
    m = np.mean(vec[:normL], 0)
    out = np.zeros(vec.shape, vec.dtype)
    if m.shape is ():
        if np.abs(m) > th:
            out = vec / m - 1.0
    else:
        zi = np.where(np.abs(m) < th)
        m[zi] = -np.inf
        out = vec / m - 1.0
    return out


from scipy import sparse


#spsolve = sparse.linalg.spsolve
def baseline_als(y, lam=None, p=0.1, niter=10):
    """Implements an Asymmetric Least Squares Smoothing
    baseline correction algorithm
    (P. Eilers, H. Boelens 2005)
    """
    L = len(y)
    if lam == None:
        lam = L**2
    D = sparse.csc_matrix(np.diff(np.eye(L), 2))
    w = np.ones(L)
    for i in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.T)
        z = sparse.linalg.spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z
