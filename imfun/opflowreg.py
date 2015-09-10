# Optical flow and image registration. Algorithms and interface to skimage, image_registration and pyimreg

from __future__ import division
import itertools as itt
from functools import partial
from collections import defaultdict

import numpy as np
from numpy import linalg


from scipy import sparse, stats
from scipy.ndimage.interpolation import map_coordinates
from scipy.interpolate import RectBivariateSpline

from skimage import feature as skfeature


from imfun import atrous,lib, fseq
#from cluster import euclidean


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

class LKP_image_aligner ():
    def __init__(self,mesh,wsize):
        self.mesh = mesh
        self.wsize = wsize
    def __call__(self, im1, im2, maxiter=10,
                 weigh_by_shitomasi = False,
                 smooth_vfield = 2):

        mesh = self.mesh
        xgrid, ygrid = np.unique(mesh[:,1]), np.unique(mesh[:,0])

        gshape = (len(xgrid), len(ygrid))
        p = np.zeros((2,)+gshape)
        imx = im1.copy()
        for niter in xrange(maxiter):
            vx = lk_opflow(imx,im2, mesh, wsize=self.wsize)

            if weigh_by_shitomasi:
                st_resp = skfeature.corner_shi_tomasi(im1)
                st_resp =  lib.clip_and_rescale(st_resp, 5000)
                weights = np.array([st_resp[tuple(l)] for l in mesh])
                vx = vx*weights[:,None]

            vfields = vx.T.reshape((2,)+gshape)

            if smooth_vfield:
                vfields = map(partial(atrous.smooth, level=smooth_vfield), vfields)

            p += vfields
            imx = self.warp_image(im1, p)
        return imx, p

    def grid_shift_coords(self, vfields, outshape):
        mesh = self.mesh
        xi,yi = np.arange(outshape[1]), np.arange(outshape[0])
        xgrid, ygrid = np.unique(mesh[:,1]), np.unique(mesh[:,0])
        dxsampler,dysampler = [RectBivariateSpline(xgrid, ygrid, vfields[dim])
                               for dim in 1,0]
        dx = dxsampler(xi,yi)
        dy = dysampler(xi,yi)
        return dx, dy

    def warp_image(self, im, vfields, mode='nearest'):
        xi,yi = np.arange(im.shape[1]), np.arange(im.shape[0])
        xii,yii = np.meshgrid(xi,yi)
        dx,dy = self.grid_shift_coords(vfields, im.shape)
        return map_coordinates(im, [yii-dy, xii-dx],mode=mode)

class GK_image_aligner:
    """
    Align two images using Lucas-Canade algorithm and Greenberg-Kerr parametrization model
    """
    def __call__(self, img, template, p0x, p0y, maxiter=100, damping=1,
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
            T = self.warp_image(template,px,py) 
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
   
# ------------ registration wrappers ------------
# TODO: make this sane and move to a proper place

def lkp_wrapper(image, template, wsize=25):
    sh = image.shape
    mstride=wsize/3
    grange = range(wsize/2,sh[0]-wsize/2,mstride) # square images FIXME
    mesh = np.array([(i,j) for i in grange for j in grange])
    aligner = LKP_image_aligner(mesh, wsize)
    _,p = aligner(image,template)
    def _regfn(coordinates):
        shifts = aligner.grid_shift_coords(p, sh)
        return [c-s for c,s in zip(coordinates, shifts[::-1])]
    return _regfn

#TODO: wrapper functions should return functions that transform pixel grid, not frame. 
# this should make function composition more stable

# https://github.com/pyimreg/imreg
from imreg import register, sampler, model

def imreg_wrapper(image, template, tform=model.Affine(), registrator=None):
    if registrator is None: 
        registrator = register.Register()
    template, image = map(register.RegisterData, (template,image))
    step, search = registrator.register(image, template, tform, sampler=sampler.bilinear)
    def _regfn(coordinates):
        #return sampler.bilinear(img, tform(step.p, register.RegisterData(img).coords).tensor)
        ir_coords = model.Coordinates.fromTensor(coordinates)
        return tform(step.p, ir_coords).tensor
    return _regfn

def sktrans_wrapper(image,template):
    shift = skfeature.register_translation(template, image,upsample_factor=16.)[0]
    def _regfn(coordinates):
        return [c - p for c,p in zip(coordinates, shift)]
    return _regfn

def gk_wrapper(image, template, nparam=11, transpose=True, constraint=20.):
    if transpose:
        template = template.T
        image = image.T
    aligner = GK_image_aligner()
    shift = skfeature.register_translation(template, image, upsample_factor=4.)[0]
    p0x,p0y = np.ones(nparam)*shift[1], np.ones(nparam)*shift[0]
    res, p = aligner(image, template, p0x,p0y, maxiter=50, constraint = constraint, corr_threshold=0.95)
    p = p0x,p0y
    def _regfn(coordinates):
        sh = coordinates[0].shape
        dx = aligner.wcoords_from_params1d(p[0], sh)
        dy = aligner.wcoords_from_params1d(p[1], sh)
        if transpose:
            dx,dy = dy,dx
        return [coordinates[0]-dy, coordinates[1]-dx]
    return _regfn

#from multiprocessing import Pool

#!conda install dill
import dill
#!pip install git+https://github.com/uqfoundation/pathos 
from pathos.multiprocessing import ProcessingPool

def parametric_warp(img, fn):
    """given an image and a function to warp coordinates,
    warp image to the new coordinates"""
    start_coordinates = np.meshgrid(*map(np.arange, img.shape))[::-1]
    return map_coordinates(img, fn(start_coordinates))

def apply_warps(frames, warps):
    """
    returns result of applying warps for given frames (one warp per frame)
    """
    out = np.array([parametric_warp(f,w) for f,w in itt.izip(frames, warps)])
    if isinstance(frames, fseq.FrameSequence):
        out = fseq.open_seq(out)
        out.meta = frames.meta
    return out

def register_stack_to_template(frames, template, regfn, njobs=4, **fnargs):
    """
    Given stack of frames (or a FSeq obj) and a template image, 
    align every frame to template and return a list of functions,
    which take an image and return warped image, aligned to template.
    """
    #return [regfn(img, template, **fnargs) for img in frames] 
    pool = ProcessingPool(nodes=njobs) 
    return pool.map(partial(regfn, template=template, **fnargs), frames)

def register_stack_recursive(frames, regfn):
    """
    Given stack of frames, 
    align frames recursively and return a mean frame of the aligned stack and
    a list of functions, each of which takes an image and return warped image, 
    aligned to this mean frame.
    """
    #import sys
    #sys.setrecursionlimit(len(frames))
    L = len(frames)
    if L < 2:
        return frames[0], [lambda f:f]
    else:
        mf_l, warps_left = register_stack_recursive(frames[:L/2], regfn)
        mf_r, warps_right = register_stack_recursive(frames[L/2:], regfn)
        fn = regfn(mf_l, mf_r)
        fm = 0.5*(parametric_warp(mf_l,fn) + mf_r)
        return fm, [lib.flcompose2(fx,fn) for fx in warps_left] + warps_right
        #return fm, [fnutils.flcompose2(fn,fx) for fx in fn1] + fn2
