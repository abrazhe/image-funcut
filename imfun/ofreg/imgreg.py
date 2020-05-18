

import numpy as np

from skimage import feature as skfeature

from . import lkp_
from . import gk_
from . import clg_
from .warps import Warp


try:
    # https://github.com/pyimreg/imreg
    import imreg.register
    import imreg.model
    import imreg.sampler # do we really need that?
    _with_imreg = True
except ImportError:
    print("Can't load imreg package, affine and homography registrations won't work")
    _with_imreg = False



_boundary_mode='nearest'
_output = 'flow' # {fn, flow, dct}



def shifts( image, template, upsample=16.):
    shift = skfeature.register_translation(template, image,upsample_factor=upsample)[0]
    def _regfn(coordinates):
        return [c - p for c,p in zip(coordinates, shift[::-1])]
    return Warp.from_function(_regfn, image.shape)

def _imreg(image, template, tform):
    if not _with_imreg:
        raise ImportError("Don't have imreg module")
    aligner = imreg.register.Register()
    sh = image.shape
    template, image = list(map(imreg.register.RegisterData, (template,image)))
    step, search = aligner.register(image, template, tform)
    def _regfn(coordinates):
        ir_coords = imreg.model.Coordinates.fromTensor(coordinates)
        out =  tform(step.p, ir_coords).tensor
        return out
    return Warp.from_function(_regfn, sh)


def affine(image,template):
    return _imreg(image, template, imreg.model.Affine())

def homography(image,template):
    return _imreg(image, template, imreg.model.Homography())


def greenberg_kerr(image, template, nparam=11, transpose=True, **fnargs):
    if transpose:
        template = template.T
        image = image.T
    aligner = gk_.GK_image_aligner()
    shift = skfeature.register_translation(template, image, upsample_factor=4.)[0]
    p0x,p0y = np.ones(nparam)*shift[1], np.ones(nparam)*shift[0]

    if not 'maxiter' in fnargs:
        fnargs['maxiter'] = 25

    res, p = aligner(image, template, p0x,p0y, **fnargs)

    sh = image.shape
    def _make_flow():
        dx, dy = [aligner.wcoords_from_params1d(pe, sh[::-1]) for pe in p]
        if transpose:
            dx,dy = dy,dx
        return -dx,-dy
    def _regfn(coordinates):
        dx, dy = _make_flow()
        return [coordinates[0]+dx, coordinates[1]+dy]
    return Warp.from_function(_regfn, sh)


def slkp( image, template, wsize=25, **fnargs):
    sh = image.shape
    aligner = lkp_.LKP_image_aligner(wsize)
    _,p = aligner(image,template, **fnargs)
    def _regfn(coordinates):
        shifts = aligner.grid_shift_coords(p, sh)
        return [c-s for c,s in zip(coordinates, shifts)]
    return Warp.from_function(_regfn,sh)

def mslkp(image, template, nl=3, wsize=25,**fnargs):
    aligner = lkp_.MSLKP_image_aligner()
    w = aligner(image,template, **fnargs)
    return Warp.from_array(w)

def msclg( image, template, nl=3, algorithm='pcgs',alpha=1e-5,
          verbose=False, **fnargs):
    aligner = clg_.MSCLG(algorithm=algorithm,verbose=verbose)
    w = aligner(image, template,nl, **fnargs)
    return Warp.from_array(w)
