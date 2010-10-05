from distutils.core import setup

setup(name='image-funcut',
      version = '0.0.6',
      scripts = ['imfun/frame_viewer.py'],
      py_modules = ['imfun.ui',
                    'imfun.pica',
                    'imfun.synthdata',
                    'imfun.opt',
                    'imfun.leica',
                    'imfun.lib',
                    'imfun.pca',
                    'imfun.fseq',
                    'imfun.fnmap',
                    'imfun.MLFImage'],)
