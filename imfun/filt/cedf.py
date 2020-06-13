## Coherence-enhancing diffusion filtering

import numpy as np
from numpy.random import randn
from numba import jit
from scipy import ndimage as ndi

import sys

from . import filt2d

Fx = np.array([(-3., 0, 3), (-10,0,10), (-3,0,3)])/32.
Fy = Fx.T


@jit
def calculate_gradient(img):
    return filt2d(img, Fx), filt2d(img, Fy)


#@jit
def calc_structure_tensor(Ix, Iy,rho=3):
    # so far no smoothing
    J = np.zeros(Ix.shape+(3,))
    J[...,0] = Ix*Ix
    J[...,1] = Ix*Iy
    J[...,2] = Iy*Iy
    #nrows, ncols = Ix.shape
    # for r in range(nrows):
    #     for c in range(ncols):
    #         dx = Ix[r,c]
    #         dy = Iy[r,c]
    #         J[r,c] = dx*dx, dx*dy, dy*dy
    if rho > 0:
        for i in range(3):
            J[...,i] = ndi.gaussian_filter(J[...,i],rho)
    return J#.reshape(nrows,ncols,2,2)


@jit
def orientations_from_structure_tensor(J):
    nrows, ncols = J.shape[:2]
    out = np.zeros((nrows,ncols,2))
    for r in range(nrows):
        for c in range(ncols):
            j11,j12,j22 = J[r,c]
            dx = (j11-j22)**2 + 4*j12**2
            dxsr = dx**0.5

            v1,v2 = 2*j12, j22-j11+dxsr
            vnorm =(v1*v1 + v2*v2)**0.5
            cosa,sina = v1/vnorm, v2/vnorm
            out[r,c] = cosa,sina
    return out

@jit
def calc_flowness(im,sigma=1.0):
    Ix, Iy = filt2d(im, Fx), filt2d(im, Fy)
    J = calc_structure_tensor(Ix, Iy, sigma)
    nrows, ncols = im.shape
    out = np.zeros((nrows,ncols))
    for r in range(nrows):
        for c in range(ncols):
            j11,j12,j22 = J[r,c]
            dx = (j11-j22)**2 + 4*j12**2
            dxsr = np.sqrt(dx)#**0.5

            mu1 = 0.5*(j11 + j22 + dxsr)
            mu2 = 0.5*(j11 + j22 - dxsr)
            rx = 0
            if (mu1+mu2) != 0:
                rx = (mu1-mu2)/(mu1+mu2)
            out[r,c] = rx
            pass
    return out


@jit
def calc_diffusion_tensor(J, c1=0.001, c2=1.0,r_cutoff=0.2):
    sh = J.shape[:2]
    #lam1 = np.zeros(sh)
    #lam2 = np.zeros(sh)
    D = np.zeros(J.shape)
    nrows, ncols = sh
    r,dxr,mu1,mu2 = 0,0,0,0
    j11,j12,j22 = 0,0,0
    lam1 = lam2 = c1
    cosa,sina = 0,1
    v1,v2,vnorm = 0,0,0
    for row in range(nrows):
        for col in range(ncols):
            j11,j12,j22 = J[row,col]
            dxr = (j11-j22)**2 + 4*j12**2
            dxsr = np.sqrt(dxr)#**0.5
            mu1 = 0.5*(j11 + j22 + dxsr)
            mu2 = 0.5*(j11 + j22 - dxsr)

            lam1 = lam2 = c1
            r = 0
            if (mu1+mu2) != 0:
                r = (mu1-mu2)/(mu1+mu2)

            if r > r_cutoff:
                lam2 = c1 + (1-c1)*np.exp(-c2/r)
            else:
                lam1 = lam2 = c1*10

            v1,v2 = 2*j12, j22-j11+dxsr
            vnorm =(v1*v1 + v2*v2)**0.5
            cosa,sina = v1/(vnorm+1e-6), v2/(vnorm +1e-6)

            a = lam1*cosa**2 + lam2*sina**2
            b = (lam1-lam2)*sina*cosa
            c = lam1*sina**2 + lam2*cosa**2
            D[row,col] = a,b,c
    return D


def coh_enh_diff_f_rhs(u,rho=3,r_cutoff=0.2):
    Ix = filt2d(u, Fx)
    Iy = filt2d(u, Fy)

    J = calc_structure_tensor(Ix,Iy,rho)
    D = calc_diffusion_tensor(J,r_cutoff=r_cutoff)

    j1 = D[...,0]*Ix + D[...,1]*Iy
    j2 = D[...,1]*Ix + D[...,2]*Iy

    return filt2d(j1,Fx) + filt2d(j2,Fy)


def coh_enh_diffusion(u,dt=0.2,T=20,verbose=True,rho=3):
    u = u.copy()
    t = 0
    while t < T:
        upd = coh_enh_diff_f_rhs(u,rho=rho)
        u += dt*upd
        t += dt
        if verbose:
            sys.stderr.write("\r model time %2.3f"%t)
    if verbose:
        sys.stderr.write('\n')
    return u

def __isotropic_diffusion(u,dt=0.2,T=20,verbose=True,rho=3):
    u = u.copy()
    t = 0
    while t < T:
        upd = coh_enh_diff_f_rhs(u,rho=rho)
        u += dt*upd
        t += dt
        if verbose:
            sys.stderr.write("\r model time %2.3f"%t)
    if verbose:
        sys.stderr.write('\n')
    return u


def adams_bashforth(rhs, init_state, dt=0.25,tstart=0, tstop=100,  fnkwargs=None):
    if fnkwargs is None:
        fnkwargs = {}

    ndim = len(init_state)
    tv = arange(tstart,tstop,dt)

    xprev = init_state
    fprev = rhs(xprev, **fnkwargs)
    xcurr = xprev + dt*fprev

    for k,t in enumerate(tv[1:-1]):
        fnew = rhs(xcurr, **fnkwargs)
        xnew = xcurr + 0.5*dt*(3*fnew - fprev)
        fprev = fnew
        xcurr = xnew
    return xnew

def coherence_shockf(img, niter=10,line_sigma=0.5):
    rho = 3*line_sigma
    u = img.copy()
    for i in range(niter):
        v = ndi.gaussian_filter(u, line_sigma)
        Ix,Iy = calculate_gradient(u)
        J = calc_structure_tensor(Ix,Iy,rho)

        omega = orientations_from_structure_tensor(J)
        cw = omega[...,0]
        sw = omega[...,1]

        vx,vy = calculate_gradient(v)
        vxx = filt2d(vx,Fx)
        vyy = filt2d(vy,Fy)
        vxy = 0.5*(filt2d(vx,Fy) + filt2d(vy,Fx))

        vww = cw**2*vxx + 2*cw*sw*vxy + sw**2*vyy

        dil = ndi.grey_dilation(u,3)
        ero = ndi.grey_erosion(u,3)

        u = np.where(vww<0,dil,ero)
    return u
