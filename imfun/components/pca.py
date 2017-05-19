import numpy as np
from numpy import array,dot,argsort,diag,where, zeros, sqrt,float,linalg
from numpy.linalg import eig, eigh, inv, norm, svd


from ..core import ah


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
    #  Normalize s, so it contains standard deviations of projections on principal axes
    s /= (ndata-1)**0.5
    ## U is eigenvectors of ``Xc Xc.H`` in columns
    ## Vh is eigenvectors of ``Xc.H Xc`` in rows
    Z = U[:,:ncomp]*(ndata-1)**0.5 # whitened data with unit variance
    ## equivalently (?)
    ## Z = dot((U/s).T[:ncomp], Xc)
    #K = Vh[:ncomp].T/s
    return Z[:,:ncomp], Vh[:ncomp], s[:ncomp], X_mean

def pca_points(X,ncomp=2):
    """Variant for ellipse fitting

    Input:
      - X : data points, dimensions are columns, independent observarions are rows
            i.e. each row is a point with x,y,... coordinates. Data can also be
            regarded as a list of coordinate tuples, e.g. [(x1,y1), (x2,y2), ...]

    Output:
      - Vh : PC vectors
      - phi: rotation of main axis (in degrees)
      - ranges: data ranges of projections on PC axes
      - s  : standard deviations
      - center: center of the data
      - Y : data projections on PCs
    """
    c0 = X.mean(axis=0)
    X1 = (X - c0)  # remove empirical mean
    U,s,Vh = svd(X1, full_matrices=False)
    # now rows of Vh are the PCs and columns of U are the coefficients
    Y = [dot(L.reshape(1,-1), X1.T) for L in Vh ]
    ranges = [y.max() - y.min() for y in Y]
    phi = np.rad2deg(np.arctan2(Vh[0,1],Vh[0,0])) # rotation of main axis (for Ellipse)
    return Vh[:ncomp], phi, ranges[:ncomp],s[:ncomp]**0.5,  c0, array(Y[:ncomp])

def pca_svd_project(X, Vh):
    c0 = X.mean(axis=0)
    X1 = (X - c0)
    return array([np.dot(L.reshape(1,-1), X1.T).reshape(-1) for L in Vh ]).T

def _whitenmat(X, ncomp=None):
    "Assumes data are nframes x npixels"
    n,p = list(map(float, X.shape))
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

try:
    from sklearn import decomposition as skd
    _with_sklearn = True
    _pca_frames_algorithm='truncated'
except ImportError:
    _with_sklearn = False
    _pca_frames_algorithm = 'svd'

class PCA_frames():
    def __init__(self,frames, npc=20, algorithm=_pca_frames_algorithm):
        self.npc = npc
        self.sh = frames[0].shape
        self.mean_frame = np.mean(frames, axis=0)
        data = ah.ravel_frames(frames-self.mean_frame)
        if algorithm == 'svd':
            u,s,vh = np.linalg.svd(data, full_matrices=False)
            self.u = u[:,:npc]
            self.s = s[:npc]
            self.vh = vh[:npc]
            self.coords = np.array([self.project(frame) for frame in frames])
        elif algorithm == 'truncated' and _with_sklearn:
            tsvd = skd.TruncatedSVD(npc)
            self.tsvd = tsvd
            self.coords = tsvd.fit_transform(data)
            self.vh = tsvd.components_
        else:
            raise InputError("Can't use algorithm %s"%algorithm)
    def project(self, frame):
        return self.vh.dot(np.ravel(frame-self.mean_frame))
    def approx(self, frame):
        return self.project(frame).dot(self.vh).reshape(self.sh) + self.mean_frame
    def rec_from_coefs(self,coefs):
        return coefs.dot(self.vh).reshape(self.sh) + self.mean_frame


#### Old stuff -------------------------------------------------------------------------------------------------------

def _pca_trick(X):
    """PCA with transformation trick"""
    ndata, ndim = X.shape
    #X_mean = X.mean(axis=-1)[:, np.newaxis]
    X_mean = X.mean(axis=0)[np.newaxis,:]
    Y = X - X_mean # remove mean
    Y = X
    e, C = eigh(dot(Y,Y.T))
    print(e)
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
    print("Please don't use this, it's not ready")
    #return
    n_data, n_dimension = X.shape # (m x n)
    Y = X - X.mean(axis=0)[np.newaxis,:] # remove mean
    #C = dot(Y, Y.T) # (n x n)  covariance matrix
    C = dot(Y.T, Y)
    print(C.shape)
    es, EV = eigh(C)  # eigenvalues, eigenvectors

    ## take non-negative eigenvalues
    non_neg, = where(es>=0)
    neg = where(es<0)
    if len(neg)>0:
        if verbose:
            print("pca1: Warning, C have %d negative eigenvalues" %len(neg))
        es = es[non_neg]
        EV = EV[:,non_neg]
    #tmp = dot(Y.T, EV).T
    #V1 = tmp[::-1]
    #S1 = sqrt(es)[::-1]
    return EV
