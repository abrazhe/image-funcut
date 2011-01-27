import numpy as np
from numpy import dot,argsort,diag,where,dstack,zeros, sqrt,float,real,linalg
#from numpy import *
from numpy.linalg import eigh, inv, eig, norm
from numpy.random import randn, rand
import time
try:
    from scipy.stats import skew
    _skew_loaded = True
    from scipy.linalg import orth
    _orth_loaded = True
except:
    _orth_loaded = False
    _skew_loaded = False

## TODOs:
## [ ] show how much variance is accounted by pcs
## [X] function to reshape back to movie
## [X] behave nicely if no scipy available (can't sort by skewness then)
## [ ] simple GUI (traits?)
## [ ] masks from ICs or PCs    

def pca1 (X):
    """
    Simple principal component analysis
    X as Npix by Nt matrix
    X should be normalized and centered beforehand
    --
    returns:
    - U, (Npix by Nesq:esq>0), matrix of PC 'filters' 
    - EV (Nt by Nesq:esq>0), matrix of PC 'signals'
     (eigenvalues of temporal covariance matrix). Signals are in columns
    - esq, vector of eigenvalues
    """
    tick = time.clock()
    Nx, Nt = X.shape
    C = dot(transp(X), X)/Nt # temporal covariance matrix

    ## who knows why do they do that in [1]
    mean_space  = X.mean(axis=0).reshape(1,-1)
    C -= dot(mean_space.T, mean_space)

    es, EV = eigh(C)  # eigenvalues, eigenvectors

    ## take non-negative eigenvalues
    non_neg, = where(es>0)
    neg = where(es<0)
    if len(neg)>0:
        print "pca1: Warning, C have %d negative eigenvalues" %len(neg)
        es = es[non_neg]
        EV = EV[:,non_neg]
    S = diag(np.sqrt(es))
    whitenmat = dot(EV, inv(S)) # whitenting matrix
    ki = argsort(es)[::-1]      # indices for sorted eigenvalues    
    U = dot(X, dot(EV,inv(S)))  # spatial filters
    print "PCA finished in %03.2f sec" %(time.clock() - tick)
    return U[:,ki], EV[:,ki], es[ki]


## Whitening

def whiten(x, U, v):
    pass
            


def cell_ica1(pc_filters, pc_signals, sev, nIC=20,
              PCuse = None,
              mu = 0.3):
    "Wrapper around fastica1, relies on previous use of pca"
    if not isinstance(PCuse, slice):
        if not PCuse :
            PCuse = slice(PCuse)
        elif len(PCuse) <= 3:
            PCuse = slice(*PCuse)

    nPCs =  pc_signals[:,PCuse].shape[1]
    pc_fuse = pc_filters[:,PCuse]
    sev_use = sev[PCuse]

    pc_suse = pc_signals - pc_signals.mean(axis=0).reshape(1,-1)
    pc_suse = pc_suse[:,PCuse]

    mux = sptemp_concat(pc_fuse, pc_suse, mu)

    ica_A,_ = fastica1(mux, nIC=nIC)
    ica_sig = dot(transp(ica_A), transp(pc_suse))

    ica_filters = transp(dot(pc_fuse,
                             dot(diag(sev_use**-0.5),ica_A)))

    if _skew_loaded:
        skewsorted = argsort(skew(ica_sig, axis=1))[::-1]
    else:
        skewsorted = range(nIC)
    return ica_filters[skewsorted].T, ica_sig[skewsorted].T


def transp(m):
    "conjugate transpose"
    return m.conjugate().transpose()


def pow2_nonlinf(X,B):
    _, siglen = X.shape
    return dot(X, dot(transp(X), B)**2.0) / siglen

def pow3_nonlinf(X,B):
    _,siglen = X.shape
    return dot(X, dot(transp(X), B)**3.0)/siglen - 3*B

def tanh_nonlinf(X,B, a1 = 1.0):
    nrows,siglen = X.shape
    ht = np.tanh(a1 * dot(transp(X), B))
    B = dot(X, ht)/siglen
    B -= dot(np.ones((nrows,1)),
             np.sum(1 - ht**2, axis=0).reshape(1,-1)) * B / siglen * a1
    return B



def fastica1(X, nIC=None, guess=None,
             nonlinfn = pow3_nonlinf,
             #nonlinfn = pow2_nonlinf,
             #nonlinfn = tanh_nonlinf,
             termtol = 5e-7, maxiters = 2e3):
    "Simplistic ICA with FastICA algorithm"
    tick = time.clock()
    nPC, siglen = X.shape
    nIC = nIC or nPC-1
    guess = guess or rand(nPC,nIC) - 0.5

    if _orth_loaded:
        guess = orth(guess)

    B, Bprev = guess, zeros(guess.shape, np.float64)
    Bold2 = zeros(guess.shape, np.float64)

    iters, minAbsCos, minAbsCos2 = 0,0,0

    errvec = []
    while (iters < maxiters) and ((1-minAbsCos2) > termtol)\
              and ((1-minAbsCos) > termtol):
        B = nonlinfn(X,B)
        ## Symmetric orthogonalization.
        ## W(W^TW)^{-1/2}
        a = dot(transp(B), B)
        a = msqrt(inv(a))
        B = dot(B, real(a))

        # Check termination condition
        minAbsCos = np.min(abs(diag(dot(transp(B), Bprev))))
        minAbsCos2 = np.min(abs(diag(dot(transp(B), Bold2))))
        Bold2 = np.copy(Bprev)
        Bprev = np.copy(B)
        errvec.append(1-minAbsCos) # history of convergence
        iters += 1

    if iters < maxiters:
        print "Success: ICA Converged in %d tries" %iters
    else:
        print "Fail: reached maximum number of iterations %d reached"%maxiters

    print "ICA finished in %03.2f sec" % (time.clock()- tick)
    return B.real, errvec

def fastica_defl(X, nIC=None, guess=None,
             nonlinfn = pow3_nonlinf,
             #nonlinfn = pow2_nonlinf,
             #nonlinfn = tanh_nonlinf,
             termtol = 5e-7, maxiters = 2e3):
    tick = time.clock()
    nPC, siglen = X.shape
    nIC = nIC or nPC-1
    guess = guess or randn(nPC,nIC)

    if _orth_loaded:
        guess = orth(guess)

    B, Bprev = zeros(guess.shape, np.float64), zeros(guess.shape, np.float64)

    iters, minAbsCos  = 0,0

    errvec = []
    icc = 0
    while icc < nIC:
        w = randn(nPC,1) - 0.5
        w -= dot(dot(B, transp(B)), w)
        w /= norm(w)

        wprev = zeros(w.shape)
        for i in xrange(long(maxiters) +1):
            w -= dot(dot(B, transp(B)), w)
            w /= norm(w)
            #wprev = w.copy()
            if (norm(w-wprev) < termtol) or (norm(w + wprev) < termtol):
                B[:,icc]  = transp(w)
                icc += 1
                break
            wprev = w.copy()
    return B.real, errvec



### Some general utility functions:
### --------------------------------
def gauss_kern(xsize, ysize=None):
    """ Returns a normalized 2D gauss kernel for convolutions """
    xsize = int(xsize)
    ysize = ysize and int(ysize) or xsize
    x, y = mgrid[-xsize:xsize+1, -ysize:ysize+1]
    g = np.exp(-(x**2/float(xsize) + y**2/float(ysize)))
    return g / g.sum()

def gauss_blur(X,size=1.5):
    return signal.convolve2d(X,gauss_kern(size),'same')

def reshape_from_movie(mov):
    l,w,h = mov.shape
    return transp(mov.reshape(l,w*h))

def reshape_to_movie(X,nrows,ncols):
    Np, Nt = X.shape
    return X.T.reshape(Nt,nrows,ncols)

def sptemp_concat(filters, signals, mu):
    if mu == 0:
        out= filters.T # spatial only
    elif mu == 1:
        out= signals.T # temporal only
    else:
        out =  np.concatenate(((1-mu)*filters.T, mu*signals.T),
                              axis = 1)
    return out / np.sqrt(1-2*mu+2*mu**2)


def DFoF(X):
    """
    Delta F / mean F normalization for each pixel
    """
    xmean = X.mean(axis=1).reshape(-1,1)
    xmean_z = where(xmean==0)
    xmean[xmean_z] = 1.0
    out =  X/xmean - 1
    out[xmean_z,:] = 0
    return out

def DFoSD(X, tslice = None):
    """
    Delta F / S.D.(F) normalization
    """
    if not isinstance(tslice, slice):
        tslice = tslice and slice(*tslice) or slice(None)
    xsd = X[:,tslice].std(axis=1).reshape(-1,1)
    xmean = X[:,tslice].mean(axis=1).reshape(-1,1)
    return (X-xmean)/xsd

def demean(X):
    """
    Remove mean over time from each pixel
    Frames are flattened and are in columns
    """
    xmean = X.mean(axis=0).reshape(1,-1)
    return X - xmean


def winvhalf(X):
    "raise matrix to power -1/2"
    e, V = eigh(X)
    return dot(V, dot(diag((e+0j)**-0.5),transp(V)))

def msqrt(X):
    e, V = eigh(X)
    return dot(V, dot(diag(((e+0j)**0.5)), transp(V)))

def mpower(M, p):
    """
    Matrix exponentiation, works for Hermitian matrices
    """
    e,EV = linalg.eigh(M)
    return dot(transp(EV),
               dot(diag((e+0j)**p), EV))


def mask4overlay(mask,colorind=0):
    """
    Put a binary mask in some color channel
    and make regions where the mask is False transparent
    """
    sh = mask.shape
    z = np.zeros(sh)
    stack = dstack((z,z,z,np.ones(sh)*mask))
    stack[:,:,colorind] = mask
    return stack


def flcompose2(f1,f2):
    "Compose two functions from left to right"
    def _(*args,**kwargs):
        return f2(f1(*args,**kwargs))
    return _
                  
def flcompose(*funcs):
    "Compose a list of functions from left to right"
    return reduce(flcompose2, funcs)
