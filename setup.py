from distutils.core import setup

setup(name='image-funcut',
      version = '0.0.10',
      scripts = ['imfun/frame_viewer.py'],
      requires = ['swan'],
      py_modules = ['imfun.bwmorph',
                    'imfun.cluster',
                    'imfun.filt',
                    'imfun.fnmap',
                    'imfun.fseq',
                    'imfun.leica',
                    'imfun.lib',
                    'imfun.opt',
                    'imfun.pca',
                    'imfun.pica',
                    'imfun.synthdata',
                    'imfun.ui',
                    'imfun.MLFImage'],)
