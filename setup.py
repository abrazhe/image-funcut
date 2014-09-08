#!/usr/bin/python
from distutils.core import setup

setup(name='image-funcut',
      version = '0.0.27',
      scripts = ['imfun/frame_viewer.py'],
      requires = ['swan'],
      py_modules = ['imfun.atrous',
                    'imfun.bwmorph',
                    'imfun.cluster',
		    'imfun.emd',
                    'imfun.filt',
                    'imfun.fnmap',
		    'imfun.fnutils',
                    'imfun.fseq',
                    'imfun.lib',
                    'imfun.leica',
                    'imfun.mes',
                    'imfun.mmt',
                    'imfun.multiscale',
                    'imfun.mvm',
                    'imfun.opt',
                    'imfun.pca',
                    'imfun.pica',
                    'imfun.som',
                    'imfun.synthdata',
		    'imfun.tiffile',
                    'imfun.track',
                    'imfun.ui',
                    'imfun.MLFImage'],)
