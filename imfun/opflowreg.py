# Optical flow and image registration. Algorithms and interface to skimage, image_registration and pyimreg

from __future__ import division
import itertools as itt
from collections import defaultdict

import numpy as np
from numpy import linalg


from scipy.ndimage.interpolation import map_coordinates
from scipy import sparse, stats

from skimage import feature as skfeature

from imfun import lib

#from numba import jit

def lk_opflow(im1, im2, locations, wsize=11, It=None, zeromean=False,
              calc_eig=False, weigh_by_eig = False):
    """
    Optical flow estimation in a set of points using Lucas-Kanade algorithm    
    """
    if zeromean:
        im1 = im1-np.mean(im1)
        im2 = im2-np.mean(im2)
    Ix,Iy = np.gradient(im1)
    if It is None:
        It = im1-im2 # reverse sign first order difference
    else:
        It = -It
    hw = np.floor(wsize/2)
    dim = len(locations[0])
    out = np.zeros((len(locations), dim+int(bool(calc_eig or weigh_by_eig))))
    for k,loc in enumerate(locations):
        window = tuple([slice(l-hw,l+hw+1) for l in loc])
        Ixw,Iyw = Ix[window].ravel(),Iy[window].ravel()
        Itw = It[window].reshape(-1,1)
        AT = np.vstack((Ixw,Iyw))
        
        ATA = AT.dot(AT.T)
        #V = pinv(ATA).dot(AT).dot(Itw)
        V  = linalg.lstsq(ATA, AT.dot(Itw))[0]
        #print V.shape
        out[k,:dim] = V.ravel()
        if calc_eig or weigh_by_eig:
            u,s,vh = linalg.svd(AT, full_matrices=False)
            out[k,-1] = (s[1]/s[0])**2
    if weigh_by_eig:
        out = out[:,:dim]*out[:,dim:]
    return out

from scipy.interpolate import RectBivariateSpline
from imfun import atrous
#from cluster import euclidean
#@jit
def lk_grid_shift_coords(im, mesh, vfields):
    xi,yi = np.arange(im.shape[1]), np.arange(im.shape[0])
    xgrid, ygrid = np.unique(mesh[:,1]), np.unique(mesh[:,0])
    dxsampler,dysampler = [RectBivariateSpline(xgrid, ygrid, vfields[dim])
                           for dim in 1,0]
    dx = dxsampler(xi,yi)
    dy = dysampler(xi,yi)
    return dx, dy

def lk_grid_warp(im, mesh, vfields):
    xi,yi = np.arange(im.shape[1]), np.arange(im.shape[0])
    xii,yii = np.meshgrid(xi,yi)
    dx,dy = lk_grid_shift_coords(im, mesh, vfields)
    return map_coordinates(im, [yii-dy, xii-dx],mode='nearest')

def lk_register_grid(im1,im2, mesh, maxiter=10, wsize=11,
                     weigh_by_shitomasi = False,
                     smooth_vfield = 5):
    
    xgrid, ygrid = np.unique(mesh[:,1]), np.unique(mesh[:,0])

    gshape = (len(xgrid), len(ygrid))
    p = np.zeros((2,)+gshape)
    imx = im1.copy()
    for niter in xrange(maxiter):
        vx = lk_opflow(imx,im2, mesh, wsize=wsize)

        if weigh_by_shitomasi:
            st_resp = skfeature.corner_shi_tomasi(im1)
            st_resp =  lib.clip_and_rescale(st_resp, 5000)
            weights = np.array([st_resp[tuple(l)] for l in mesh])
            vx = vx*weights[:,None]

        vfields = vx.T.reshape((2,)+gshape)

        if smooth_vfield:
            vfields = map(lambda f:atrous.smooth(f,level=smooth_vfield), vfields)

        p += vfields
        imx = lk_grid_warp(im1, mesh, p)
        
    return imx, p
        
    



class GK_image_aligner:
    """
    Align two images using Lucas-Canade algorithm and Greenberg-Kerr parametrization model
    """
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

    def warp_image(self, img, dx,dy, mode='nearest'):
        sh = img.shape
        xi,yi = np.meshgrid(np.arange(sh[1]), np.arange(sh[0]))
        return map_coordinates(img, [yi+ dy, xi+dx], mode=mode)
    
    def warp_image_parametric(self, img, px,py):
        dx = self.wcoords_from_params1d(px, img.shape)
        dy = self.wcoords_from_params1d(py, img.shape)
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
   
    def align_image(self, img, template, p0x, p0y, maxiter=100, damping=1,
                    constraint = 10., # max allowed shift in px
                    corr_threshold = 0.99, 
                    dp_threshold = 1e-3):
        #for iterc in range(maxiter):
        
        n = len(p0x)
        blocksize,remd = self.get_blocksize(n,img.shape)
        dDdp = self.calc_dDdp(n, img.shape)
        
        px,py = p0x,p0y
        
        acc =  defaultdict(lambda:list())
        
        for niter in range(maxiter):
            T = self.warp_image_parametric(template,px,py) 
            gTy,gTx = map(np.ravel, np.gradient(T))
            
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
