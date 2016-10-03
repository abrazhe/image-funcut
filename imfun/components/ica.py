import numpy as np
from numpy import array, dot, diag, zeros
from numpy.linalg import eig, eigh, inv, norm, svd
from numpy.random import randn

try:
    from scipy.stats import skew
    _skew_loaded = True
    from scipy.linalg import orth
    _orth_loaded = True
    from scipy import signal
except ImportError:
    _orth_loaded = False
    _skew_loaded = False


from ..fseq import ravel_frames
from ..fseq import shape_frames

from . import pca
from .pca import whitenmat2


def st_ica(X, ncomp = 20,  mu = 0.2, npca = None, reshape_filters=True):
    """Spatiotemporal ICA for sequences of images

    Input:
      - `X` -- list of 2D arrays or 3D array with first axis = time
      - `ncomp` -- number of components to resolve
      - `mu`  -- weight of temporal input, :math:`mu = 0 -> spatial`; mu = 1 -> temporal
      - `npca` -- number of principal components to calculate (default, equals
        to the number of independent components
      - `reshape_filters` -- if true, `ICA` filters are returned as a sequence
        of images (3D array, Ncomponents x Npx x Npy)

    Output:
      - ica_filters, ica_signals
    """
    data = reshape_from_movie(X) # nframes x npixels
    sh = X[0].shape

    npca = (npca is None) and ncomp or npca

    ## note, svd of transposed data is slightly faster
    #pc_f, pc_s, ev = _whitenmat(data)
    pc_f, pc_s, ev = whitenmat2(data.T)

    pc_f = pc_f[:npca]
    pc_s = pc_s[:npca]
    ev = ev[:npca]

    mux = sptemp_concat(pc_f[:npca], pc_s[:npca], mu)

    _, W = fastica(mux, ncomp=ncomp, whiten=False)
    ica_signals = dot(W, pc_s)
    ## I doubt we really need this scaling at all
    #a = diag(1.0/np.sqrt(ev[:ncomp])) # do we need sqrt?
    #ica_filters = dot(dot(a, W), pc_f)
    ica_filters = dot(W, pc_f)

    if _skew_loaded:
	skews = array([skew(f.ravel()) for f in ica_filters])
	skew_signs = np.sign(skews)
        skewsorted = np.argsort(np.abs(skews))[::-1]
	for fnumber,s in enumerate(skew_signs):
	    ica_signals[fnumber] *= s
	    ica_filters[fnumber] *= s
    else:
        skewsorted = range(ncomp)
    if reshape_filters:
        ica_filters = reshape_to_movie(ica_filters[skewsorted], sh)
    else:
        ica_filters = ica_filters[skewsorted]
    return ica_filters, ica_signals[skewsorted]


def transp(m):
    "conjugate transpose"
    return m.conjugate().transpose()

def sptemp_concat(filters, signals, mu):
    if mu == 0:
        out= filters # spatial only
    elif mu == 1:
        out= signals # temporal only
    else:
        out =  np.concatenate(((1-mu)*filters, mu*signals),
                              axis = 1)
    return out / np.sqrt(1-2*mu+2*mu**2)


pow3nonlin = {'g':lambda X: X**3,
              'gprime': lambda X: 3*X**2}

pow3nonlinx = {'g':lambda X,args: X**3,
              'gprime': lambda X,args: 3*X**2}

logcoshnonlin = {'g': lambda X: np.tanh(X),
                 'gprime': lambda X: 1.0 - np.tanh(X)**2}

def _sym_decorrelate(X):
    ":math:`W <- W \\cdot (W^T \\cdot W)^{-1/2}`"
    a = dot(X, transp(X))
    ev, EV = eigh(a)

    return dot(dot(dot(EV, np.diag(1.0/np.sqrt(ev))),
                   EV.T),
               X)

def _ica_symm(X, nIC=None, guess=None,
              nonlinfn = pow3nonlin,
              termtol = 5e-7, max_iter = 2e3,
              verbose=False):
    "Simplistic ICA with FastICA algorithm"
    nPC, siglen = map(np.float, X.shape)
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
            print "Fail: reached maximum number of iterations %d reached"%max_iter
    return B.real



def fastica(X, ncomp=None, whiten = True,
            algorithm = 'symmetric',
            nonlinfn = pow3nonlin,
            tol = 1e-04, max_iter = 1e3, guess = None):
    """Fast ICA algorithm realisation.

    Input:
     - X -- data matrix with observations in rows (Nsamples x Nfeatures)
     - ncomp -- number of components to resolve  [all possible]
     - whiten -- whether to whiten the input data
     - nonlinfn -- nonlinearity function [pow3nonlin]
     - tol -- finalisation tolerance, [1e-04]
     - max_iter -- maximal number of iterations [1000]
     - guess -- initial guess [None]

    Output:
     - S -- estimated sources (in rows)
     - W -- unmixing matrix
    """
    n,p = map(np.float, X.shape)
    if whiten:
        XW, Uh, s, _ = pca.pca(X, ncomp) # whitened data and projection matrix
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
    nPC, siglen = X.shape
    nIC = nIC or nPC-1
    guess = guess or randn(nPC,nIC)

    if _orth_loaded:
        guess = orth(guess)

    B = zeros(guess.shape, np.float64)

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
    e,EV = eigh(M)
    return dot(transp(EV),
               dot(diag((e+0j)**p), EV))
