from distutils.core import setup

setup(name='image-funcut',
      version = '0.0.12',
      scripts = ['imfun/frame_viewer.py'],
      requires = ['swan'],
      py_modules = ['imfun.atrous',
                    'imfun.bwmorph',
                    'imfun.cluster',
                    'imfun.filt',
                    'imfun.fnmap',
                    'imfun.fseq',
                    'imfun.lib',
                    'imfun.leica',
                    'imfun.mvm'
                    'imfun.opt',
                    'imfun.pca',
                    'imfun.pica',
                    'imfun.som',
                    'imfun.synthdata',
                    'imfun.ui',
                    'imfun.MLFImage'],)
