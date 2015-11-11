#!/usr/bin/python
#from distutils.core import setup

from setuptools import setup 
#import path
import subprocess

#here = path.abspath(path.dirname(__file__))
#version = '0.1.4'

def version_from_git():
    try:
        proc = subprocess.Popen(["git", "describe", "--tags"], stdout=subprocess.PIPE)
        out,_ = proc.communicate()
        version = out.strip().replace('-','+')
    except Exception:
    	version = "0.unspecified"
    return version


setup(name='image-funcut',
      # version = version_from_git(), # I have problems with this so far
      version = '0.2.22',
      url = "https://github.com/abrazhe/image-funcut",
      author = "Alexey Brazhe",
      license = "GPL",
      description = "View, analyse and transform dynamic imaging data",
      scripts = ['scripts/funcut.py', 'scripts/stabilize-framesequence.py'],
      packages = ['imfun'],
      data_files = [('Examples/image-funcut',['scripts/stab-params-example.json'])],
      install_requires = ['numpy',
                          'scipy',
                          'numba>=0.20.0',
                          'scikit-image>=0.11.0',
                          'imreg',
                          'dill>=0.2.4',
                          'pathos>=0.1',
                          'h5py>=2.5.0',
                          'pandas>=0.16.0',
                          'traits>=4.4.0',
                          'traitsui>=4.4.0',
                          'wxpython',
                          'swan>=0.6.7'],
      dependency_links=['https://github.com/pyimreg/imreg/archive/master.zip#egg=imreg',
                        'https://github.com/uqfoundation/dill/archive/master.zip#egg=dill',
                        'https://github.com/uqfoundation/pathos/archive/master.zip#egg=pathos>=0.2'],
      classifiers = [
          'Development Status :: 4 - Beta',
          "Intended Audience :: Science/Research",
          'License :: OSI Approved :: GNU General Public License (GPL)',
          'Operating System :: OS Independent :: Linux',
          'Programming Language :: Python :: 2.7',
          'Topic :: Scientific/Engineering',
        ],
      )

