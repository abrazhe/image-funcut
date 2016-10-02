#### --- jPCA ------------------------------
#### See Churchland, Cunningham, Kaufman, Foster, Nuyujukian, Ryu, Shenoy. 
#### Neural population dynamics during reaching Nature. 2012 Jul 5;487(7405):51-6. 
#### doi: 10.1038/nature11129

import numpy as np
from numpy import array,  zeros, ndim
from numpy.linalg import eig, eigh,  norm,  lstsq


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
    M0 = lstsq(X,dX)[0] # general unconstrained L2-good solution
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
        out = zeros(n*(n-1)/2)
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
