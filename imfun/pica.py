## Principal and independent components 

import numpy as np
from numpy import dot,argsort,diag,where, zeros, sqrt,float,linalg
from numpy import sign, ndim, array
from numpy import mgrid
#from numpy import *
from numpy.linalg import eig, eigh, inv, norm, svd
from numpy.random import randn

try:
    from scipy.stats import skew
    _skew_loaded = True
    from scipy.linalg import orth
    _orth_loaded = True
    from scipy import signal
except:
    _orth_loaded = False
    _skew_loaded = False

## TODOs:
## [X] show how much variance is accounted for by pcs
## [X] function to reshape back to movie
## [X] behave nicely if no scipy available (can't sort by skewness then)
## [ ] simple GUI (traits?)
## [ ] masks from ICs or PCs    


def pca(X, ncomp=None):
    """PCA decomposition via SVD

    Input:
      - X -- an array (Nsamples x Kfeatures)
        this arrangement is faster if there are many samples with low number of feature
        it is also more 'pythonic', as X[n] would be n-th data point
    Output:
      - Z -- whitened data (Nsamples x ncomp), projection on first ncomp PC components
      - Vh -- (ncomp, Kfeatures) ncomp principal vectors (right eigenvectors of XX.T
      - s -- eigenvalues
      - X_mean -- sample mean
      Note that whitening matrix should be calculated as Vh^T \cdot diag(1/s)
    """
    ndata, ndim = X.shape
    X_mean = X.mean(0).reshape(1,-1)
    Xc = X - X_mean # remove mean (center data)
    U,s,Vh = svd(Xc, full_matrices=False)
    ## U is eigenvectors of ``Xc Xc.H`` in columns
    ## Vh is eigenvectors of ``Xc.H Xc`` in rows
    Z = U[:,:ncomp] # whitened data
    ## equivalently (?)
    ## Z = dot((U/s).T[:ncomp], Xc)
    #K = Vh[:ncomp].T/s
    return Z[:,:ncomp], Vh[:ncomp], s[:ncomp], X_mean

def pca_points(X):
    """Variant for ellipse fitting

    Input:
      - X : data points, dimensions are columns, independent observarions are rows
            i.e. each row is a point with x,y,... coordinates. Data can also be
            regarded as a list of coordinate tuples, e.g. [(x1,y1), (x2,y2), ...]
    
    Output:
      - Vh : PC vectors
      - phi: rotation of main axis (in degrees)
      - ranges: data ranges of projections on PC axes
      - center: center of the data
      - Y : data projectins on PCs
    """
    c0 = X.mean(axis=0)
    X1 = (X - c0)  # remove empirical mean
    U,s,Vh = svd(X1, full_matrices=False)
    Y = [dot(L.reshape(1,-1), X1.T) for L in Vh ]
    ranges = [y.max() - y.min() for y in Y]
    phi = np.rad2deg(np.arctan2(Vh[0,1],Vh[0,0])) # rotation of main axis (for Ellipse)
    return Vh, phi, ranges, c0, array(Y)

def pca_svd_project(X, Vh):
    c0 = X.mean(axis=0)
    X1 = (X - c0)
    return array([np.dot(L.reshape(1,-1), X1.T).reshape(-1) for L in Vh ]).T
    
def _whitenmat(X, ncomp=None):
    "Assumes data are nframes x npixels"
    n,p = map(float, X.shape)
    Xc = X - X.mean(axis=-1)[:, np.newaxis]
    U, s, Vh = svd(Xc, full_matrices=False)
    #K = (U/s).T[:ncomp] # fixme: do I really have to scale by s?
    #Z  = np.dot(K,Xc)
    Z = Vh[:ncomp]  # (the upper variant is equivalent
    return Z, U.T[:ncomp], s[:ncomp]

def whitenmat2(X):
    """
    Input: array
    Assumes data X is an npixels x nframes matrix


    Output:
      - pc_filters
      - pc_signals
      - singular values

    both pc_filters and pc_signals are arranged in rows,
    i.e. pc_filters[0] is the first filter, pc_signals[1] is the second signal,
    etc. This will be handy later for concatenation of spatial and temporal components
    """
    ## curious, which axis it's best to subtract mean from?
    ## currently I subtract the mean signal from all signals because the mean
    ## frame is already supposed to be subtracted from the movie
    Xc = X - X.mean(axis=0)
    #Xc = X - X.mean(axis=-1)[:, np.newaxis]
    pc_filters, s, pc_signals = svd(Xc, full_matrices=False)
    return pc_filters.T, pc_signals, s



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
        skewsorted = argsort(np.abs(skews))[::-1]
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



pow3nonlin = {'g':lambda X: X**3,
              'gprime': lambda X: 3*X**2}

pow3nonlinx = {'g':lambda X,args: X**3,
              'gprime': lambda X,args: 3*X**2}

logcoshnonlin = {'g': lambda X: np.tanh(X),
                 'gprime': lambda X: 1.0 - np.tanh(X)**2}

def _sym_decorrelate(X):
    ":math:`W <- W \\cdot (W^T \\cdot W)^{-1/2}`"
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
    n,p = map(float, X.shape)
    if whiten:
        XW, Uh, s, _ = pca(X, ncomp) # whitened data and projection matrix
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

#### --- jPCA ------------------------------
#### See Churchland, Cunningham, Kaufman, Foster, Nuyujukian, Ryu, Shenoy. 
#### Neural population dynamics during reaching Nature. 2012 Jul 5;487(7405):51-6. 
#### doi: 10.1038/nature11129

from scipy import optimize as opt

## TODOs:
## [ ] Normalize vectors in jPCs
## [ ] Allow for sorting according to most variance explained, not frequency
## [ ] Better documentation
## [ ] Examples
## [ ] Note usage in Kalman filtering

def jpca(X, npc=12, symm=-1,verbose=False):
    """
    Find jPCA vector pairs
    Input:
    ------
        - X -- array of shape (Nsamplex x Kfeatures) 
        - npcs -- number of principal components to use [12]
        - symm -- search for skew-symmetric solution if symm-1 or 
          for symmetric solution if symm=1
    Output:
    -------
        - jPCs : array of jPCA vector pairs sorted from highest frequency 
                 rotations (oscillations) to lower frequency oscillations 
        - Vh : PC vectors 
        - s : PC singular values
        - Xc: data center, which was substracted prior to PCA

    """
    U, Vh, s, Xc = pca(X, npc)
    dU = np.diff(U, axis=0)
    Mskew = skew_symm_solve(dU, U[:-1], symm=symm,verbose=verbose)
    evals, evecs = eig(Mskew)
    jPCs = [make_eigv_real(evecs[:,p]) for p in 
            by_chunks(xrange(len(evals)))]
    return array(jPCs), Vh, s, Xc

def make_eigv_real(p_vec, p_vals=None):
    "given a pair of complex conjugate eigenvectors, return two orthogonal vectors t "
    return array([p_vec[:,0]+p_vec[:,1], 1j*(p_vec[:,0]-p_vec[:,1])]).T.real

def skew_symm_solve(dX, X, symm=-1, sp_lambda = 0,verbose=False):
    """
    Find jPCA solution 1st pass 
    dX,X : (nt, ncomp) matrices (FIXME)"""
    M0 = linalg.lstsq(X,dX)[0] # general unconstrained L2-good solution
    #M0 = np.random.randn(X.shape[1],X.shape[1])
    M0k = 0.5 *(M0 + symm*M0.T) # skew/symm component
    m0 = reshape_skew(M0k, symm)
    #sp_lambda = 0.1
    def skew_objective_fn(x):
        M = reshape_skew(x, symm)
        DXM = dX - X.dot(M)
        f = norm(DXM)**2 + sp_lambda*np.sum(np.abs(x))
        D = (DXM).T.dot(X)
        df = 2*reshape_skew(D-D.T)
        return f, df
    res = opt.minimize(skew_objective_fn, m0, method='L-BFGS-B', jac=True, 
                             options = {'maxiter':500})
    #####res = opt.fmin_l_bfgs_b(skew_objective_fn, m0, approx_grad=True,)
    #res = opt.minimize(skew_objective_fn, m0)
    if verbose:
        print 'Optimization', res.success and 'success' or 'failure'
    return reshape_skew(res.x,symm)


_small_number = 1e-8
def reshape_skew(m, s=-1):
    """reshape n(n-1)/2 vector into nxn skew symm matrix or vice versa
    indices in m are in row-major order (it's Python, babe)
    when s == -1, returns skew-symmetric matrix 
        M = -M.T
    when s == 1, returns symmetric matrix
        M = M.T
    """
    s = sign(s)
    if ndim(m) == 1: # m is a vector, we need a matrix
        n = 0.5*(1 + np.sqrt(1 + 8*len(m)))
        if not (n==round(n)):
            print "length of m doesn't lead to a square-sized matrix of size n(n-1)/2"
            return
        n = int(n)
        out = zeros((n,n))
        ind_start = 0
        for i in xrange(n):
            out[i,i+1:] = m[ind_start:ind_start+n-i-1]
            ind_start += n-i-1
        out += s*out.T
    elif ndim(m) == 2: # m is a matrix, we need a vector
        if not np.equal(*m.shape):
            print "matrix m is not square"
            return
        if (norm(m - s*m.T)) > _small_number:
            print "matrix m is not skew-symmetric or symmetric"
            return
        n = m.shape[0]
        out = np.zeros(n*(n-1)/2)
        ind_start = 0
        for i in range(n):
            out[ind_start:ind_start+n-i-1] = m[i,i+1:]
            ind_start += n-i-1
    return out

def by_chunks(seq, n=2):
    count, acc = 0, []
    for s in seq:
        count +=1
        acc.append(s)
        if not (count%n):
            yield acc
            count, acc = 0, []

### Some general utility functions:
### --------------------------------------------------------------------------------------

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

def reshape_to_movie(X,(nrows,ncols)):
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


#### Old stuff -------------------------------------------------------------------------------------------------------

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

    returns:
    - EV (Nt by Nesq:esq>0), matrix of PC 'signals'
     (eigenvalues of temporal covariance matrix). Signals are in columns
    - esq, vector of eigenvalues
    """
    print "Please don't use this, it's not ready"
    #return
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

