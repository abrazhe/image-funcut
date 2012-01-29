import numpy as np
from numpy import dot,argsort,diag,where,dstack,zeros, sqrt,float,real,linalg
#from numpy import *
from numpy.linalg import eigh, inv, eig, norm, svd
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
## [ ] show how much variance is accounted for by pcs
## [X] function to reshape back to movie
## [X] behave nicely if no scipy available (can't sort by skewness then)
## [ ] simple GUI (traits?)
## [ ] masks from ICs or PCs    


def pca(X, ncomp=None):
    """PCA decomposition via SVD
    Input
    ~~~~~~
    X -- an array where each column contains observations from one probe
         and each row is different probe (dimension)

    Output
    ~~~~~~~
    Z -- whitened matrix
    K -- PC matrix
    s -- eigenvalues
    X_mean -- sample mean
    """
    ndata, ndim = X.shape
    X_mean = X.mean(axis=-1)[:,np.newaxis]    
    Xc = X - X_mean # remove mean
    U,s,Vh = svd(Xc, full_matrices=False)
    ## U is eigenvectors of ``Xc Xc.H`` in columns
    ## Vh is eigenvectors of ``Xc.H Xc`` in rows
    Z = Vh[:ncomp] # whitened data
    ## equivalently,
    ## Z = dot((U/s).T[:ncomp], Xc)
    K = (U/s).T[:ncomp]
    return Z, K, s[:ncomp], X_mean

def pca_svd(X):
    """Variant for ellipse fitting
    Input:
    X : data points, dimensions are columns, independent observarions are rows

    Output:
    Vh : PC vectors
    phi: rotation of main axis (in degrees)
    ranges: data ranges of projections on PC axes
    center: center of the data
    """
    c0 = X.mean(axis=0)
    X1 = (X - c0)
    U,s,Vh = svd(X1, full_matrices=False)
    ax, ay = Vh.T
    Y = [dot(L.reshape(1,-1), X1.T) for L in Vh ]
    ranges = [y.max() - y.min() for y in Y]
    phi = np.rad2deg(np.arctan(Vh[0,1]/Vh[0,0])) # rotation of main axis (for Ellipse)
    return Vh, phi, ranges, c0

def st_ica(X, ncomp = 20,  mu = 0.3, npca = None, reshape_filters=True):
    """Spatiotemporal ICA for sequences of images
    Input:
    ~~~~~~
    X -- list of 2D arrays or 3D array with first axis = time
    ncomp -- number of components to resolve
    mu [0.3] -- spatial vs temporal, mu = 0 -> spatial; mu = 1 -> temporal
    npca [None] -- number of principal components to calculate (default, equals
                   to the number of independent components
    reshape_filters [True] -- if true, ICA filters are returned as a sequence
                              of images (3D array, Ncomponents x Npx x Npy)
    
    Output:
    ~~~~~~~
    ica_filters, ica_signals
    """
    data = reshape_from_movie(X) # nframes x npixels
    sh = X[0].shape

    npca = (npca is None) and ncomp or npca
    
    pc_f, pc_s, ev = whitenmat(data, npca)
    #pc_f, pc_s, ev, _ = whitenmat(data, ncomp)

    mux = sptemp_concat(pc_f, pc_s, mu)

    _, W = fastica(mux, ncomp=ncomp, whiten=False)
    ica_sig = dot(W, pc_s)
    a = diag(1.0/np.sqrt(ev[:ncomp]))
    ica_filters = dot(dot(a, W), pc_f)

    if _skew_loaded:
        skewsorted = argsort(skew(ica_sig, axis=1))[::-1]
    else:
        skewsorted = range(ncomp)
    if reshape_filters:
        ica_filters = reshape_to_movie(ica_filters[skewsorted], *sh)
    else:
        ica_filters = ica_filters[skewsorted]
    return ica_filters, ica_sig[skewsorted]


## Whitening

def whitenmat(X, ncomp=None):
    n,p = map(float, X.shape)
    Y = X - X.mean(axis=-1)[:, np.newaxis]
    U, s, Vh = svd(Y, full_matrices=False)
    K = (U/s).T[:ncomp]
    return np.dot(K, Y), K, s[:ncomp]



def transp(m):
    "conjugate transpose"
    return m.conjugate().transpose()



pow3nonlin = {'g':lambda X: X**3,
              'gprime': lambda X: 3*X**2}

pow3nonlinx = {'g':lambda X,args: X**3,
              'gprime': lambda X,args: 3*X**2}

logcoshnonlin = {'g': lambda X: np.tanh(X),
                 'gprime': lambda X: 1.0 - np.tanh(X)**2}

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

    
    guess = guess or np.random.normal(size=(nIC,nPC))
    guess = _sym_decorrelate(guess)

    B, Bprev = zeros(guess.shape, np.float64), guess

    iters, errx = 0,termtol+1
    g, gp = nonlinfn['g'], nonlinfn['gprime']


    while (iters < max_iter) and (errx > termtol):
        bdotx = dot(Bprev, X)
        gwtx = g(bdotx)
        gp_wtx = gp(bdotx)#/siglen
        B = dot(gwtx, X.T)/siglen - dot(diag(gp_wtx.mean(axis=1)), Bprev)
        B = _sym_decorrelate(B)
        errx = max(abs(abs(diag(dot(B, Bprev.T)))-1))
        Bprev = np.copy(B)
        iters += 1
    if verbose:
        if iters < max_iter:
            print "Success: ICA Converged in %d tries" %iters
        else:
            print "Fail: reached maximum number of iterations %d reached"%maxiters
    return B.real


def fastica(X, ncomp=None, whiten = True,
            algorithm = 'symmetric',
            nonlinfn = pow3nonlin,
            tol = 1e-04, max_iter = 1e3, guess = None):
    """Fast ICA algorithm realisation.
    Input:
    ~~~~~~
    X -- data matrix with observations in rows
    ncomp -- number of components to resolve  [all possible]
    whiten -- whether to whiten the input data [True]
    nonlinfn -- nonlinearity function [pow3nonlin]
    tol -- finalisation tolerance, [1e-04]
    max_iter -- maximal number of iterations [1000]
    guess -- initial guess [None]

    Output:
    ~~~~~~~
    S -- estimated sources (in rows)
    W -- unmixing matrix
    """
    n,p = map(float, X.shape)
    if whiten:
        XW, Uh, s, _= pca(X, ncomp) # whitened data and projection matrix
        #XW, Uh, _ = whitenmat(X, ncomp) # whitened data and projection matrix
    else:
        XW = X.copy()
    XW *= np.sqrt(p)
    kwargs = {'nIC':ncomp, 'termtol':tol, 'nonlinfn':nonlinfn,
              'max_iter':max_iter, 'guess':guess}
    algorithms = {'symmetric':_ica_symm,
                  'deflation':None}
    fun = algorithms.get(algorithm, 'symmetric')
    W  = fun(XW, **kwargs)
    if whiten:
        S = dot(dot(W,Uh),X)
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
    Nt, Np = X.shape
    return X.reshape(Nt,nrows,ncols)

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

#### Old stuff

def _pca_trick(X):
    """PCA with transformation trick"""
    ndata, ndim = X.shape
    #X_mean = X.mean(axis=-1)[:, np.newaxis]
    X_mean = X.mean(axis=0)[np.newaxis,:]    
    Y = X - X_mean # remove mean
    Y = X
    e, C = eigh(dot(Y,Y.T))
    print e
    V = dot(Y.T, C)
    return dot(V, inv(diag(sqrt(e)))), sqrt(e), X_mean



def _pca1 (X, verbose=False):
    """
    Simple principal component decomposition (PCA)
    X as Npix by Nt matrix
    X should be normalized and centered beforehand
    --
    returns:
    - EV (Nt by Nesq:esq>0), matrix of PC 'signals'
     (eigenvalues of temporal covariance matrix). Signals are in columns
    - esq, vector of eigenvalues
    """
    print "Please don't use this, it's not ready"
    #return
    tick = time.clock()
    n_data, n_dimension = X.shape # (m x n)
    Y = X - X.mean(axis=0)[np.newaxis,:] # remove mean
    #C = dot(Y, Y.T) # (n x n)  covariance matrix
    C = dot(Y.T, Y)
    print C.shape
    es, EV = eigh(C)  # eigenvalues, eigenvectors

    ## take non-negative eigenvalues
    non_neg, = where(es>=0)
    neg = where(es<0)
    if len(neg)>0:
        if verbose:
            print "pca1: Warning, C have %d negative eigenvalues" %len(neg)
        es = es[non_neg]
        EV = EV[:,non_neg]
    #tmp = dot(Y.T, EV).T
    #V1 = tmp[::-1]
    #S1 = sqrt(es)[::-1]
    return EV

