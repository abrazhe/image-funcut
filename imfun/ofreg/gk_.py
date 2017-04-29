

import itertools as itt

from collections import defaultdict

import numpy as np

from scipy import sparse, stats
from scipy.ndimage.interpolation import map_coordinates


class GK_image_aligner:
    """
    Align two images using Lucas-Canade algorithm and Greenberg-Kerr parametrization model
    """
    def __call__(self, img, template, p0x, p0y, maxiter=100, damping=1,
                    constraint = 10., # max allowed shift in
                    corr_threshold = 0.99,
                    dp_threshold = 1e-3):
        #for iterc in range(maxiter):

        n = len(p0x)
        blocksize,remd = self.get_blocksize(n,img.shape)
        dDdp = self.calc_dDdp(n, img.shape)

        px,py = p0x,p0y

        acc =  defaultdict(lambda:list())

        for niter in range(maxiter):
            T = self.warp_image(template,px,py)
            gTy,gTx = list(map(np.ravel, np.gradient(T)))

            dDdp2 = np.vstack([dDdp*gTx, dDdp*gTy])

            diff_img = (img - T).ravel()
            C = (dDdp2 * diff_img).sum(-1)

            H = np.zeros((n*2,n*2))
            for i in np.arange(n-1):
                tsl = slice(i*blocksize,(i+1)*blocksize)
                for ia,ib in itt.product(*[(i, i+1, i+n, i+n+1)]*2):
                    H[ia,ib] += (dDdp2[ia,tsl]*dDdp2[ib,tsl]).sum(-1)

            H = sparse.csr_matrix(H)
            dp = sparse.linalg.spsolve(H,C)
            #dp = linalg.lstsq(H,C)[0]  # can do wierd things in last parameters
            #dp = linalg.solve(H,C)     # bad result
            #dp = linalg.pinv(H).dot(C)  # best result so far
            dpx,dpy = dp[:n],dp[n:]

            acc['d'].append(sum(diff_img**2))
            acc['rho'].append(stats.pearsonr(img.ravel(), T.ravel())[0])
            mdp = np.amax(abs(dp))
            acc['mdp'].append(mdp)
            px = px + damping*dpx
            py = py + damping*dpy
            px = np.clip(px, -constraint, constraint)
            py = np.clip(py, -constraint, constraint)
            #damping *= damping

            if (acc['rho'][-1] > corr_threshold) or (mdp < dp_threshold):
                break

        return acc, (px,py)

    def get_blocksize(self, nparams, shape):
        N = np.prod(shape)
        return N//(nparams-1), N%(nparams-1)

    def wcoords_from_params1d(self,p, shape):
        Npx = np.prod(shape)
        D = np.zeros(Npx)
        nparams = len(p)
        blocksize, remd = self.get_blocksize(nparams, shape)
        for i in np.arange(nparams-1):
            tdur = blocksize + (i > nparams-3)*remd
            tv = np.arange(tdur,dtype=int)
            D[i*blocksize:i*blocksize+tdur] = p[i] + (p[i+1]-p[i])*tv/tdur
        return D.reshape(shape)

    def warp_image(self, img, px,py, mode='nearest'):
        sh = img.shape
        dx = self.wcoords_from_params1d(px, sh)
        dy = self.wcoords_from_params1d(py, sh)
        xi,yi = np.meshgrid(np.arange(sh[1]), np.arange(sh[0]))
        return map_coordinates(img, [yi+ dy, xi+dx], mode=mode)

        return self.warp_image(img,dx,dy)

    def calc_dDdp(self, nparams, shape):
        #todo: sparse matrices
        Npx = np.prod(shape)
        dDdp = np.zeros((nparams,Npx))
        blocksize,remd= self.get_blocksize(nparams, shape)
        tv = np.arange(Npx)/blocksize
        for k in np.arange(nparams-1):
            tkp1 = k + 1
            if k >= nparams-2:
                tkp1+=remd/blocksize
            tk = k
            trange = ((tk <= tv)*(tv < tkp1))>0
            dDdp[k,:] += (1 - (tv-tk))*trange
            dDdp[k+1,:] += (tv-tk)*trange
        return dDdp
