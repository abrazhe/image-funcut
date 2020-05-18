# Algorithm by Bruhn, Weickert, Schnorr


import sys

import numpy as np

from numpy import array

from scipy import ndimage
from scipy.ndimage.interpolation import map_coordinates

from numba import jit
from functools import partial

from .warps import Warp

from imfun.multiscale import pyramid_from_zoom
from imfun.filt import nearestpd,mirrorpd,filt2d,dctsplines
from imfun.filt import l2spline_thresholded,l2spline,l1spline

_boundary_mode='nearest'

# TODO: rethink, what should be class attributes,
#       and what should be __call__ arguments

def copy_to_larger_cpad(source, destination):
    nr,nc = source.shape
    nrd,ncd = destination.shape
    destination[:min(nr,nrd),:min(nc,ncd)] = source
    if np.any(np.array(destination.shape)>source.shape):
        destination[nr:,:nc] = source[nr-1,:][None,:]
        destination[:nr,nc:] = source[:,nc-1][:,None]
        destination[nr:,nc:] = source[-1,-1]


class MSCLG(object):
    def __init__(self,
                 output = 'flow',
                 algorithm='pcgs',
                 sor_omega=1.9,
                 niter=None,
                 reltol = 1e-4,
                 verbose=False):
        self.output = output
        self.reltol = reltol
        self.algorithm=algorithm
        self.niter=niter
        self.omega = 1.9
        self.kstop = 10
        self.verbose=verbose
    def __call__(self, source, target, nl=None, alpha=1e-5, rho=10., rho_l1 = 0.,
                 do_dct_regularization=False,
                 dct_regularization_upto=50,
                 dct_regularization_threshold=5,
                 dct_regularization_smooth=None,
                 sigma=0.,  wt=1, correct_alpha=True):
        if nl is None:
            nl = int(np.log2(np.max(source.shape)/32))
        if sigma > 0:
            target = ndimage.gaussian_filter(target,sigma)
            source = ndimage.gaussian_filter(source,sigma)

        if correct_alpha:
            low = np.min([target,source])
            high = np.max([target,source])
            alpha *= abs(high-low)

        pt = pyramid_from_zoom(target,nl)
        ps = pyramid_from_zoom(source,nl)

        u = [np.zeros_like(p) for p in pt]
        v = [np.zeros_like(p) for p in pt]
        clg_args = dict(alpha=alpha,rho=rho,wt=wt)
        for level in range(nl-1,-1,-1):
            h = 2.0**level
            if level < nl-1:
                psx = Warp.from_array((u[level],v[level]))(ps[level], mode=_boundary_mode)
            else:
                psx = ps[level]
            #clg_args['rho'] =  rho/h
            (ui,vi),cerr = self.clg_of(psx,pt[level], **clg_args)
            u[level] += ui
            v[level] += vi
            if level > 0:
                sh = u[level-1].shape
                sl = tuple(slice(s) for s in sh)
                ux = ndimage.zoom(u[level],2, mode=_boundary_mode)[sl]*2
                vx = ndimage.zoom(v[level],2, mode=_boundary_mode)[sl]*2
                copy_to_larger_cpad(ux,u[level-1])
                copy_to_larger_cpad(vx,v[level-1])
        # L1-smoothing
        if rho_l1 > 0:
            u[0] = dctsplines.l1spline(u[0],rho_l1)
            v[0] = dctsplines.l1spline(v[0],rho_l1)

        if do_dct_regularization:
            if dct_regularization_smooth is None:
                rho_l2 = np.max(source.shape)/4
            else:
                rho_l2 = dct_regularization_smooth
            u[0] = l2spline_thresholded(u[0],rho_l2,
                                        nharmonics=dct_regularization_upto,
                                        th = dct_regularization_threshold)
            v[0] = l2spline_thresholded(v[0],rho_l2,
                                        nharmonics=dct_regularization_upto,
                                        th = dct_regularization_threshold)
        if self.output is 'full':
            return (u[0], v[0]), cerr
        else:
            return (u[0],v[0])

    def clg_of(self, source, target, alpha=1e-5, rho=10., wt=1):


        if self.niter is None:
            self.niter = np.prod(source.shape)/8.

        It = source-target

        kdx1 = array([[-0.5, 0, 0.5]])
        kdx2 = array([[1,-8,0,8,-1]])/12.0
        kdx3 = array([[-1,9,-45,0,45,-9,1]])/60.0


        derfilt = kdx2

        Ix1,Iy1 = filt2d(target,derfilt),filt2d(target,derfilt.T,)
        Ix2,Iy2 = filt2d(source,derfilt),filt2d(source,derfilt.T,)

        Ix = wt*Ix1 + (1-wt)*Ix2
        Iy = wt*Iy1 + (1-wt)*Iy2

        u = np.zeros_like(target)
        v = np.zeros_like(target)

        h = 1.0
        h2a = (h**2)/alpha

        J11, J12, J22 = Ix*Ix, Ix*Iy, Iy*Iy
        J13, J23 = Ix*It, Iy*It

        if rho > 0:
            smoother = lambda _f: ndimage.gaussian_filter(_f,rho)
            #smoother = lambda _f: l2spline(_f, rho)
            #smoother = partial(dctsplines.l2spline,s=rho)
            #smoother = partial(l1spline, s=rho)
            J11,J12,J13,J22,J23 = list(map(smoother, (J11,J12,J13,J22,J23)))

        J11,J12,J13,J22,J23 = [_f*h2a for _f in (J11,J12,J13,J22,J23)]

        nrows,ncols = source.shape

        cerr = np.zeros(int(self.niter))

        prev_err = 1e9
        conv_err = 0

        if self.algorithm == 'sor':
            relax_step = lambda u,v: sor_update(u,v,self.omega,J11,J12,J13,J22,J23)
        elif self.algorithm == 'pcgs':
            relax_step = lambda u,v: pcgs_update(u,v,J11,J12,J13,J22,J23)

        stopcount =0
        for ni in range(int(self.niter)):
            #cerr[ni] = sor_update(u,v,omega,J11,J12,J13,J22,J23)
            conv_err = relax_step(u,v)
            cerr[ni] = conv_err
            if ni > 0:
                if cerr[ni] < self.reltol:
                    stopcount += 1
                else:
                    stopcount = 0
                if stopcount > self.kstop:
                    if self.verbose:
                        sys.stderr.write('\rConverged at iter #%d with err %f             '%(ni, cerr[ni]))
                    break
                prev_err = conv_err

            #cerr[ni] = np.mean(np.abs(Ix*u + Iy*v + It))
            if self.verbose and not ni%100:

                sys.stderr.write('\r #iteration %d, error %f'%(ni,cerr[ni]))

        return (u,v), array(cerr[:ni+1])


@jit
def pcgs_update(u,v,J11,J12,J13,J22,J23):
    nrows,ncols = u.shape
    #nrows2,ncols2 = v.shape
    #print("Shapes!: {}, {}".format(u.shape, v.shape))
    J21 = J12
    changed = 0
    #M = np.array([[J11+4, J12], [J21, J22+4]])
    for r in range(1,nrows-1):
        for c in range(1,ncols-1):
            l = (r,c)
            M = (J11[l]+4.0,  J12[l],
                 J21[l],      J22[l]+4.0)

            g = (nhood_sum(u,l) - J13[l],
                 nhood_sum(v,l) - J23[l])

            # Solve M*[u,v].T = g.T using Cramer's rule
            Mdet = M[0]*M[3] - M[1]*M[2]

            if (np.abs(Mdet) > 1e-12):
                uXdet = g[0]*M[3] - g[1]*M[1]
                u_up = uXdet/Mdet
                #ulocal = (1-omega)*u[l] + omega*u_up
                ulocal = u_up

                vXdet = M[0]*g[1] - M[2]*g[0]
                v_up = vXdet/Mdet
                #vlocal = (1-omega)*v[l] + omega*v_up
                vlocal = v_up

                changed += (u[l]-ulocal)**2 + (v[l]-vlocal)**2
                #print shape(ulocal), shape(vlocal)
                u[l] = ulocal
                v[l] = vlocal
            else:
                changed += sor_at(u,v,l,1.0,J11,J12,J13,J22,J23)

    changed += boundary_conditions(u)
    changed += boundary_conditions(v)
    return (changed/(nrows*ncols))**0.5

@jit
def sor_update(u,v,omega,J11,J12,J13,J22,J23):
    nrows,ncols = u.shape
    changed = 0
    J21 = J12
    for r in range(1,nrows-1):
        for c in range(1,ncols-1):
            l = (r,c)
            changed += sor_at(u,v,l,omega,J11,J12,J13,J22,J23)
    changed += boundary_conditions(u)
    changed += boundary_conditions(v)
    return (changed/(nrows*ncols))**0.5

@jit
def sor_at(u,v,l,omega,J11,J12,J13,J22,J23):
    J21 = J12
    changed = 0
    u_up =  (nhood_sum(u,l) - (J12[l]*v[l] + J13[l]))/(4 + J11[l])
    ulocal = (1-omega)*u[l] + omega*u_up
    changed += (u[l]-ulocal)**2
    u[l] = ulocal

    v_up =  (nhood_sum(v,l) - (J21[l]*u[l] + J23[l]))/(4 + J22[l])
    vlocal = (1-omega)*v[l] + omega*v_up
    changed += (v[l]-vlocal)**2
    v[l] = vlocal
    return changed

@jit
def boundary_conditions(m):
    nrows,ncols = m.shape
    # first and last rows
    error = 0
    error += np.sum((m[0,:]-m[1,:])**2)
    error += np.sum((m[-1,:]-m[-2,:])**2)
    error += np.sum((m[:,0]-m[:,1])**2)
    error += np.sum((m[:,-1]-m[:,-2])**2)
    # Rows and columns:
    m[0,:] = m[1,:]
    m[-1,:] = m[-2,:]
    m[:,0] = m[:,1]
    m[:,-1] = m[:,-2]
    # corners
    m[0,0] = m[1,1]
    m[0,-1] = m[1,-2]
    m[-1,0] = m[-2,1]
    m[-1,-1] = m[-2,-2]
    return error

@jit
def nhood_sum(m, loc):
    Nr,Nc = m.shape
    out = 0
    nhood = ((-1,0), (1,0), (0,-1), (0,1))
    r,c = loc
    for i,j in nhood:
        rx = nearestpd(r+i,Nr)
        cx = nearestpd(c+j,Nc)
        out += m[rx,cx]
    return out
