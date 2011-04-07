from distutils.core import setup

setup(name='image-funcut',
      version = '0.0.9',
      scripts = ['imfun/frame_viewer.py'],
      requires = ['swan'],
      py_modules = ['imfun.ui',
                    'imfun.pica',
                    'imfun.synthdata',
                    'imfun.opt',
                    'imfun.leica',
                    'imfun.lib',
                    'imfun.pca',
                    'imfun.cluster',
                    'imfun.fseq',
                    'imfun.fnmap',
                    'imfun.bwmorph',
                    'imfun.MLFImage'],)
