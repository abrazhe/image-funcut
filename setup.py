#!/usr/bin/python
#from distutils.core import setup

from setuptools import setup 
import path
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
      version = version_from_git(),
      url = "https://github.com/abrazhe/image-funcut",
      author = "Alexey Brazhe",
      license = "GPL",
      description = "View, analyse and transform dynamic imaging data",
      scripts = ['scripts/funcut.py', 'scripts/stabilize-framesequence.py'],
      install_requires = ['numpy','scipy','swan>=0.6.7'],
      packages = ['imfun'],
      classifiers = [
          'Development Status :: 4 - Beta',
          "Intended Audience :: Science/Research",
          'License :: OSI Approved :: GNU General Public License (GPL)',
          'Operating System :: OS Independent :: Linux',
          'Programming Language :: Python',
          'Topic :: Scientific/Engineering',
        ],
      )

