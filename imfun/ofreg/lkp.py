from __future__ import division

from functools import partial

import numpy as np
from numpy import linalg

from scipy.interpolate import RectBivariateSpline
from scipy.ndimage.interpolation import map_coordinates

from skimage import feature as skfeature

from imfun import atrous, lib



class LKP_image_aligner ():
    def __init__(self,mesh,wsize):
        self.mesh = mesh
        self.wsize = wsize
    def __call__(self, source, target, maxiter=10,
                 weigh_by_shitomasi = False,
                 smooth_vfield = 2):

        mesh = self.mesh
        xgrid, ygrid = np.unique(mesh[:,1]), np.unique(mesh[:,0])

        gshape = (len(ygrid), len(xgrid))
        p = np.zeros((2,)+gshape)
        imx = source.copy()
        for niter in xrange(maxiter):
            vx = lk_opflow(target,imx, mesh, wsize=self.wsize)

            if weigh_by_shitomasi:
                st_resp = skfeature.corner_shi_tomasi(source)
                st_resp =  lib.clip_and_rescale(st_resp, 5000)
                weights = np.array([st_resp[tuple(l)] for l in mesh])
                vx = vx*weights[:,None]

            vfields = -vx.T.reshape(p.shape)
            #print 'vfields shape:', vfields.shape
            if smooth_vfield:
                vfields = map(partial(atrous.smooth, level=smooth_vfield), vfields)

            p += vfields
            imx = self.warp_image(source, p)
        return imx, p

    def grid_shift_coords(self, vfields, outshape):
        mesh = self.mesh
        xi,yi = np.arange(outshape[1]), np.arange(outshape[0])
        xgrid, ygrid = np.unique(mesh[:,1]), np.unique(mesh[:,0])
        #print '---[grid_shift_coords] outshape', outshape
        #print '---[grid_shift_coords] vfields.shape', vfields.shape
        dxsampler,dysampler = [RectBivariateSpline(ygrid, xgrid, vfields[dim]) for dim in 1,0]
        dx = dxsampler(yi,xi)
        dy = dysampler(yi,xi)
        #print '---[grid_shift_coords], dx.shape:',dx.shape,
        return dx, dy

    def warp_image(self, im, vfields, mode='nearest'):
        xi,yi = np.arange(im.shape[1]), np.arange(im.shape[0])
        xii,yii = np.meshgrid(xi,yi)
        #print '--[warp_image] vfields shape:', vfields.shape
        dx,dy = self.grid_shift_coords(vfields, im.shape)
        #print '--[warp_image] shapes: ', dx.shape, dy.shape, xi.shape, xii.shape

        return map_coordinates(im, [yii-dy, xii-dx],mode=mode)


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
    hw = int(np.floor(wsize/2))
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

    
