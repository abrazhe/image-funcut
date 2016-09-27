## Mulstiscal decomposition and reconstruction routins

import numpy as np
from scipy import ndimage

import atrous
import mmt

_dtype_ = np.float32


_boundary_mode = 'nearest'

sigmaej_starlet = [[0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],   # 0D
                   [7.235e-01, 2.854e-01, 1.779e-01, 1.222e-01, 8.581e-02, 6.057e-02,  4.280e-02, 3.025e-02, 2.138e-02, 1.511e-02, 1.067e-02, 7.512e-03], #1D
                   [0.890, 0.201, 0.086, 0.042, 0.021, 0.010, 0.005],   # 2D
                   [0.956, 0.120, 0.035, 0.012, 0.004, 0.001, 0.0005]]  # 3D


def threshold_w(coefs, th, neg=False, modulus=True, soft=False, sigmaej=atrous.sigmaej):
    """Return support for wavelet coefficients that are larger than threshold.

    Parameters:
      - coefs : wavelet coefficients
      - th : (`num` or `iterable`) -- threshold. If a number, this number is
        used as a threshold (but is scaled usign the ``sigmaej`` table for
	different levels). If a 1D array, different thresholds are used for
	different levels. If a 2D array, at each level retain only coefficients
	that are within bounds provided as columns.
      - neg: (`Bool`) -- if `True` keep coefficients that are *smaller* than
        the threshold
      - modulus: (`Bool`) -- if `True`, absolute value of coefficients is
        compared to the threshold
      - soft: (`Bool`) -- if `True` do "soft" thresholding

    Returns:
      - a list of supports (`False`--`True` masks) for each level
    """
    out = []
    nd = len(coefs[0].shape)
    fn = neg and np.less or np.greater
    for j,w in enumerate(coefs[:-1]):
	sj= sigmaej[nd][j]

	if np.iterable(th): t = th[j]
	else: t = th

	if modulus: wa = np.abs(w)
	else: wa = w

	if np.iterable(t):
	    out.append((wa > t[0]*sj)*(wa<=t[1]*sj))
	else:
	    mask = fn(wa, t*sj)
	    if soft:
		out.append(1.0*mask*np.sign(w)*(np.abs(w)-t*sj))
	    else:
		out.append(mask)
    out.append(np.ones(coefs[-1].shape)*(not neg))
    return out

def merge_supports(supp1, supp2):
    l1,l2 = len(supp1), len(supp2)

    nlevels = max(l1,l2)

    sh = [nlevels] + list(supp1[0].shape)
    out = np.zeros(tuple(sh))
    for k in range(nlevels):
        if k < l1:
            out[k] += supp1[k]
        if k < l2:
            out[k] += supp2[k]
    return out


def represent_support(supp):
    """Create a graphical representation of the support"""
    out = [2**(j+1)*supp[j] for j in range(len(supp)-1)]
    return np.sum(out, axis=0)


def simple_rec(coefs, supp=None, level=None):
    """
    Return reconstruction from wavelet coefficients and a support.
    Only coefficients where supp is non-zero are used for reconstruction.
    """
    if supp is not None:
        coefs = [c*s for c,s in zip(coefs, supp)]
    return np.sum(coefs[-1:level:-1], axis=0)

def simple_rec_iterative(coefs, supp=None, niter=5,
                         dec_fn = atrous.decompose,
                         fullout = False,
                         positive_only = True,
                         step_size=1,
                         step_damp=0.9):
    """
    Iteratively reconstruct object from wavelet coefficients and a support.
    Only coefficients where supp is non-zero are used for reconstruction.
    t.b.c.
    """

    nlevels = len(coefs)-1

    Xn = simple_rec(coefs,supp)
    if fullout:
        out = [Xn]
    for i in range(niter):
        arn = dec_fn(Xn,nlevels)
        upd = simple_rec((coefs - arn),supp)

        alpha = np.sum(upd**2)/np.sum(dec_fn(upd,nlevels)**2)
        #print ss

        Xnp1 = Xn + alpha*step_size*upd
        #Xnp1 = Xn + step_size*upd
        step_size *= step_damp

        if positive_only:
            Xnp1 *= Xnp1>=0
        Xn = Xnp1
        if fullout:
            out.append(Xn)
    if not fullout:
        out = Xn
    return out


def pyramid_from_atrous(img, nscales=4,shift=0):
    coefs = atrous.decompose(img, nscales-1 + shift)
    out = []
    approx = coefs[-1]
    for k in range(nscales-1+shift,shift,-1):
        sub = approx[::2**(k-shift),::2**(k-shift)]
        out.append(sub)
        approx += coefs[k-1]
    out.append(approx)
    return out[::-1]

def pyramid_from_zoom(img,nscales=3, scale_factor=0.5, mode=_boundary_mode):
    out = [img]
    sigma_0 = 0.6
    for i in range(nscales-1):
        sigma_zoom = sigma_0*(1.0/scale_factor**2 - 1)**0.5
        im = ndimage.gaussian_filter(out[-1], sigma_zoom)
        out.append(ndimage.zoom(im,scale_factor,mode=_boundary_mode))
    return out


## Default spline wavelet scaling function
_phi_ = np.array([1./16, 1./4, 3./8, 1./4, 1./16], _dtype_)

def qmf(filt = _phi_):
    """Quadrature mirror relationship"""
    L = len(filt)
    return [(-1)**(l+1)*filt[L-l-1] for l in range(len(filt))]
