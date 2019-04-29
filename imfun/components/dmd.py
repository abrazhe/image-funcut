import numpy as np

def omega_approx(beta):
    return 0.56*beta**3 - 0.95*beta**2 + 1.82*beta + 1.43

def svht(sv, sh):
    m,n = sh
    if m>n: 
        m,n=n,m
    omg = omega_approx(m/n)
    return omg*np.median(sv)

def min_ncomp(sv,sh):
    th = svht(sv,sh)
    return sum(sv >=th)

def dmdf_new(X,Y=None, r=None,sort_explained=False):
    if Y is None:
        Y = X[:,1:]
        X = X[:,:-1]
    U,sv,Vh = np.linalg.svd(X,False)
    if r is None:
        r = min_ncomp(sv, X.shape) + 1
    sv = sv[:r]
    V = Vh[:r].conj().T
    Uh = U[:,:r].conj().T
    B = Y@V@(np.diag(1/sv))
    
    Atilde = Uh@B
    lam, W = np.linalg.eig(Atilde)
    Phi = B@W
    #print(Vh.shape)
    # approx to b
    def _bPOD(i):
        alpha1 =np.diag(sv[:r])@Vh[:r,i]
        return np.linalg.lstsq(Atilde@W,alpha1,rcond=None)[0]
    #bPOD = _bPOD(0)
    stats = (None,None)
    if sort_explained:
        #proj_dmd = Phi.T.dot(X)
        proj_dmd = np.array([_bPOD(i) for i in range(Vh.shape[1])])
        dmd_std = proj_dmd.std(0)
        dmd_mean = abs(proj_dmd).mean(0)
        stats = (dmd_mean,dmd_std)
        kind = np.argsort(dmd_std)[::-1]
    else:
        kind = np.arange(r)[::-1] # from slow to fast
    Phi = Phi[:,kind]
    lam = lam[kind]
    #bPOD=bPOD[kind]
    return lam, Phi#,bPOD,stats
