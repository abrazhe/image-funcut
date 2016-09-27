from __future__ import division

import numpy as np

from skimage import feature as skfeature

_boundary_mode='nearest'

from . import lkp as lkpmod
from . import gk
from . import warps

try:
    # https://github.com/pyimreg/imreg
    import imreg.register
    import imreg.model
    import imreg.sampler # do we really need that?
    _with_imreg = True
except ImportError:
    print "Can't load imreg package, affine and homography registrations won't work"
    _with_imreg = False



def shifts(image, template):
    shift = skfeature.register_translation(template, image,upsample_factor=16.)[0]
    def _regfn(coordinates):
        return [c - p for c,p in zip(coordinates, shift[::-1])]
    return _regfn

def _imreg(image, template, tform):
    if not _with_imreg:
        raise NameError("Don't have imreg module")
    aligner = imreg.register.Register()
    template, image = map(imreg.register.RegisterData, (template,image))
    step, search = aligner.register(image, template, tform)
    def _regfn(coordinates):
        ir_coords = imreg.model.Coordinates.fromTensor(coordinates)
        out =  tform(step.p, ir_coords).tensor
        print out.shape
        return out
    return _regfn


def affine(image,template):
    return _imreg(image, template, imreg.model.Affine())

def homography(image,template):
    return _imreg(image, template, imreg.model.Homography())


def greenberg_kerr(image, template, nparam=11, transpose=True, **fnargs):
    if transpose:
        template = template.T
        image = image.T
    aligner = gk.GK_image_aligner()
    shift = skfeature.register_translation(template, image, upsample_factor=4.)[0]
    p0x,p0y = np.ones(nparam)*shift[1], np.ones(nparam)*shift[0]

    if not 'maxiter' in fnargs:
        fnargs['maxiter'] = 25

    res, p = aligner(image, template, p0x,p0y, **fnargs)
    def _regfn(coordinates):
        sh = coordinates[0].shape
        dx = aligner.wcoords_from_params1d(p[0], sh)
        dy = aligner.wcoords_from_params1d(p[1], sh)
        if transpose:
            dx,dy = dy,dx
        return [coordinates[0]-dx, coordinates[1]-dy]
    return _regfn


def softmesh(image, template, wsize=25, **fnargs):
    sh = image.shape

    aligner = lkpmod.LKP_image_aligner(wsize)
    _,p = aligner(image,template, **fnargs)
    def _regfn(coordinates):
        shifts = aligner.grid_shift_coords(p, sh)
        return [c-s for c,s in zip(coordinates, shifts)]
    return _regfn

def mslkp(image, template, nl=3, wsize=25,**fnargs):
    aligner = lkpmod.MSLKP_image_aligner()
    w = aligner(image,template, **fnargs)
    def _regfn(coordinates):
        return [c-s for c,s in zip(coordinates, w)]
    return _regfn



#from warps import apply_warp, flow_from_fn
