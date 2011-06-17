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

def pca_svd(X):
    ndata, ndim = X.shape
    Y = X - X.mean(axis=-1)[:, np.newaxis] # remove mean
    U,s,Vh = np.linalg.svd(X, full_matrices=False)
    Vh = Vh[:ndata] #only makes sense to return the first num_data
    return U,Vh,s

def pca1 (X, verbose=False):
    """
    Simple principal component analysis
    X as Npix by Nt matrix
    X should be normalized and centered beforehand
    --
    returns:
    - EV (Nt by Nesq:esq>0), matrix of PC 'signals'
     (eigenvalues of temporal covariance matrix). Signals are in columns
    - esq, vector of eigenvalues
    """
    tick = time.clock()
    Nx, Nt = X.shape
    Y = X - X.mean(axis=-1)[:, np.newaxis] # remove mean
    C = dot(Y, Y.T)/Nt # temporal covariance matrix

    es, EV = eigh(C)  # eigenvalues, eigenvectors

    ## take non-negative eigenvalues
    non_neg, = where(es>=0)
    neg = where(es<0)
    if len(neg)>0:
        if verbose:
            print "pca1: Warning, C have %d negative eigenvalues" %len(neg)
        es = es[non_neg]
        EV = EV[:,non_neg]
    #S = diag(np.sqrt(es))
    #whitenmat = dot(EV, inv(S)) # whitenting matrix
    ki = argsort(es)[::-1]      # indices for sorted eigenvalues    
    #U = dot(X, whitenmat)  # spatial filters
    if verbose:
        print "PCA finished in %03.2f sec" %(time.clock() - tick)
    return EV[:,ki], es[ki]

# note: returns EVs stored in columns, e.g.
# [[x1, x2, ... xn],
#  [y1, y2, ... yn],
#  ...]

## Whitening


def whiten_mat1(X):
    EV,es = pca1(X)
    S = diag(np.sqrt(es))
    return dot(EV, inv(S))

def st_ica(X, ncomp = 20,  mu = 0.3):
    """Spatiotemporal ICA for sequences of images
    Input:
    X -- list of 2D arrays or 3D array with first axis = time
    mu -- spatial vs temporal, mu = 0 -> spatial; mu = 1 -> temporal
    """

    data = reshape_from_movie(X) # nframes x npixels
    sh = X[0].shape
    
    pc_filters, pc_signals, ev = whiten_svd(data, ncomp)
    
    mux = sptemp_concat(pc_filters, pc_signals, mu)

    _, W = fastica(mux, whiten=False)
    ica_sig = dot(W, pc_signals)
    ica_filters = dot(dot(diag(1.0/np.sqrt(ev)), W), pc_filters)

    if _skew_loaded:
        skewsorted = argsort(skew(ica_sig, axis=1))[::-1]
    else:
        skewsorted = range(nIC)
    return ica_filters[skewsorted], ica_sig[skewsorted]


def transp(m):
    "conjugate transpose"
    return m.conjugate().transpose()



pow3nonlin = {'g':lambda X: X**3,
              'gprime': lambda X: 3*X**2}

pow3nonlinx = {'g':lambda X,args: X**3,
              'gprime': lambda X,args: 3*X**2}


def _sym_decorrelate(X):
    "W <- W \cdot (W^T \cdot W)^{-1/2}"
    a = dot(X, transp(X))
    ev, EV = linalg.eigh(a)
    
    return dot(dot(dot(EV, np.diag(1.0/np.sqrt(ev))),
                   EV.T),
               X)

def _ica_symm(X, nIC=None, guess=None,
              nonlinfn = pow3nonlin,
              termtol = 5e-7, max_iter = 2e3,
              verbose=False):
    "Simplistic ICA with FastICA algorithm"
    nPC, siglen = map(float, X.shape)
    nIC = nIC or nPC

    guess = guess or np.random.normal(size=(nPC,nPC))
    guess = _sym_decorrelate(guess)

    B, Bprev = zeros(guess.shape, np.float64), guess

    iters, errx = 0,termtol+1
    g, gp = pow3nonlin['g'], pow3nonlin['gprime']

    while (iters < max_iter) and (errx > termtol):
        bdotx = dot(Bprev, X)
        gwtx = g(bdotx)
        gp_wtx = gp(bdotx)/siglen
        B = dot(gwtx, X.T)/siglen - dot(np.diag(gp_wtx.mean(axis=1)), Bprev)
        B = _sym_decorrelate(B)
        errx = max(abs(abs(np.diag(dot(B, Bprev.T)))-1))
        Bprev = np.copy(B)
        iters += 1
    if verbose:
        if iters < max_iter:
            print "Success: ICA Converged in %d tries" %iters
        else:
            print "Fail: reached maximum number of iterations %d reached"%maxiters
    return B.real

def whiten_svd(X, ncomp=None):
    n,p = map(float, X.shape)
    Y = X - X.mean(axis=-1)[:, np.newaxis]
    u, s, vh = linalg.svd(X, full_matrices=False)
    K = (u/s).T[:ncomp]
    return np.dot(K, X), K, s[:ncomp]

def fastica(X, ncomp=None, whiten = True,
            algorithm = 'symmetric',
            tol = 1e-3, max_iter = 1e3, guess = None):
    n,p = map(float, X.shape)
    if whiten:
        XW, K, _ = whiten_svd(X, ncomp) # whitened and projected data
    else:
        XW = X.copy()
    XW *= np.sqrt(p)
    kwargs = {'termtol':tol, 'nonlinfn':pow3nonlin,
              'max_iter':max_iter, 'guess':guess}
    algorithms = {'symmetric':_ica_symm, 'deflation':None}
    fun = algorithms.get(algorithm, 'symmetric')
    W  = fun(XW, **kwargs)
    if whiten:
        S = dot(dot(W,K),X)
    else:
        S = dot(W,X)
    return S, W

def fastica_defl(X, nIC=None, guess=None,
             nonlinfn = pow3nonlin,
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
    return mov.reshape(l,w*h)

def reshape_to_movie(X,nrows,ncols):
    Np, Nt = X.shape
    return X.T.reshape(Nt,nrows,ncols)

def sptemp_concat(filters, signals, mu):
    if mu == 0:
        out= filters # spatial only
    elif mu == 1:
        out= signals # temporal only
    else:
        out =  np.concatenate(((1-mu)*filters, mu*signals),
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
