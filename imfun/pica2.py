### Aimed at translation of fastICA (from Octave version)
### So far, not really used anywhere in the rest of the code

import numpy as np
from numpy import dot,argsort,diag,where,dstack,zeros, sqrt,float,real
from numpy import linalg
#from numpy import *
from numpy.linalg import eigh, inv
from numpy.random import randn
import time
try:
    from scipy.stats import skew
    _skew_loaded = True
    from scipy.linalg import orth
    _orth_loaded = True
except:
    _orth_loaded = False
    _skew_loaded = False


def fastica(mixedsig, approach='symm', nIC=None,
            nonlin = 'pow3', eps = 1e-5, maxIter = 1e6,
            verbose = True, firstEig = 1, lastEig = None):
    E, D = pcamat(mixedsig, fistEig, lastEig, verbose)
    whitesig, whiteningM, dewhiteningM = whitenv(mixedsig, E, D, verbose)
    dim = whitesig.shape[0]

    if nIC is None:
        nIC = dim
    if nIC > dim:
        nIC = dim

    A, W = fpica(whitesig, whiteningM, dewhiteningM,
                 approach=approach, nIC=nIC, nonlin=nonlin,
                 eps=eps)
    icasig = dot(W, mixedsig)
    return icasig

def transp(m):
    return m.conjugate().transpose()


def pcamat(vectors, first=None, last=None, verbose=False):
    "A translation from pcamat.m from fastICA"
    nrows, ncols = vectors.shape
    C = np.cov(transp(vectors))*(ncols-1)/ncols # Covariance matrix
    max_last_eig = svdrank(C, 1e-9)

    D, E = np.linalg.eig(C) # eigenvalues, eigenvectors
    ki = argsort(D)[::-1]   # sorting vector
    if verbose:
        print E.shape
    return E, D

def whitenv(vectors, E, D, verbose=False):
    "Whitenv vectors"
    whM = dot(inv(np.sqrt(D)), transp(E))
    dewhM = dot(E, np.sqrt(D))
    return dot(whM, vectors), whM, dewhM

def fpica(X, whiteningM, dewhiteningM, approach='sym', nIC = None,
          nonlin='pow3', eps = 1e-5, maxIter = 1e6, verbose=False):
    """
    Example
    ---------

    E, D = pcamat(vectors)
    nv, wm, dwm = whitenv(vectors, E, D)
    A, W = fpica(nv, wm, dwm)
    
    """
    if np.any(X.imag):
        print "Error: input contains imaginary part"
        return None
    pass



def pow2_nonlinf(X,B):
    _, siglen = X.shape
    return dot(X, dot(transp(X), B)**2.0) / siglen

def pow3_nonlinf(X,B):
    _,siglen = X.shape
    return dot(X, dot(transp(X), B)**3.0)/siglen - 3*B

def tanh_nonlinf(X,B, a1 = 1.0):
    nrows,siglen = X.shape
    ht = np.tanh(a1 * dot(X.T, B))
    B = dot(X, ht)/siglen
    B -= dot(np.ones((nrows,1)),
             np.sum(1 - ht**2, axis=0).reshape(1,-1)) * B / siglen * a1
    return B


nonlinfndict= {'pow2':pow2_nonlinf, 'pow3':pow3_nonlinf}


def svdrank(m, tol=None):
    "Like rank in Octave"
    eps =  np.finfo(float).eps
    
    u,s,vh = linalg.svd(m)
    if tol is None:
        tol = np.max(m.shape) * s[0] * eps
    return np.sum(s>tol)
