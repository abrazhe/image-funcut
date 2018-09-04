# Image-funcut

In 2017–2018 supported by RSF grant 17-74-20089

## Overview

The `image-funcut` project is kind of a sandbox or testbed for utilities to
view, analyse and transform two-photon microscopy data or any other series of
images.

At the moment, the project includes the following `Python` modules:
  - `imfun.atrous`: comprises functions for à trous wavelet transform and
  related utilities. (Synonims: starlet transform, stationary wavelet
  transform, non-decimated 
  wavelet transform). Besides transform, there are utility functions to smooth
  input data with B-splines, remove trends in data or enhance data by noise
  suppression. 
  - `imfun.bwmorph`: helper functions for black-white morphology and binary
    masks
  - `imfun.cluster`: naive implementations of a few clustering algorithms and
    distance functions
  - `imfun.emd`: a stub for empirical mode decomposition functions
  - `imfun.filt`: various filters
  - `imfun.fnmap`: collection of functions which project XYT data to 2D images
    in various non-trivial ways
  - `imfun.fnutils`: a few functional programming-inspired utils
  - `imfun.fseq`: a keystone module: Class definitions and functions to read
    from files of several formats and represent sequences of images (both lazy
    and not) and operations on them.
  - `imfun.lib`: miscellaneous helper functions
  - `imfun.leica`: parsing XML files produced by Leica Software during export
  - `imfun.mes`: reading MES files, as created by a Femtonics microscope
  - `imfun.mmt`: multiscale median transform and hybrid median/starlet
    transform 
  - `imfun.multisale`: working with multiscale supports for starlet and
    median/starlet transforms, including iterative reconstruction from
    significant coefficients
  - `imfun.mvm`: an implementation of the Multiscale Vision Model object
    detection algorithm
  - `imfun.opt`: a stub for optimization functions
  - `imfun.pca`: unused, example PCA
  - `imfun.pica`: PCA and ICA implementations
  - `imfun.som`: implementation of Self-organizing Kohonen maps clustering
    algorithm
  - `imfun.synthdata`: functions to create simple synthetic data sets should be
    collected here
  - `imfun.tiffile`: Tiffile library by Christoph Gohlke
  - `imfun.track`: functions to track objects in a changing environment or
    align frames should be collected here
  - `imfun.ui`: Picker class -- a matplotlib-based backend-independent user
    interface to operate on `fseq` instances
  - `imfun.MLFImage`: interface to load MLF files produced by Moor laser
    speckle imaging device.
  - `frame_viewer.py`: a Traits-based GUI wrapper over `imfun.ui` and other
    modules  with additional features

One of the motivations to start this project was a *func*tional
programming approach to image data analysis, hence the name. Also, it's like a
final-cut, but with some (geeky) fun.

## Installation
### Linux and Mac, with Anaconda

First, install dependencies:
```bash
conda install pandas numba dill matplotlib h5py cython
conda install scipy scikit-image pygments 
conda install traits traitsui 

pip install https://github.com/pyimreg/imreg/archive/master.zip  
pip install https://github.com/uqfoundation/pathos/archive/master.zip
pip install swan

```

Next, if you don't want to use version-controlled source, run:

```bash
pip install https://github.com/abrazhe/image-funcut/archive/develop.zip
```

Alternatively, clone the `develop` branch of `image-funcut` from https://github.com/abrazhe/image-funcut and run:
```bash
pip install -e .
```

## Example usage
The following will load a series of TIFF files with all color channels and
start and interface to pick up ROIs, etc.
```python
    import imfun
    fs = imfun.fseq.from_tif("/path/to/many/tiff/files/*.tif",ch=None)
    p = imfun.ui.Picker(fs)
    p.start()
```
Documenting all the features is a work in progress...


## Dependencies

The project of course relies on the usual core numeric `Python` packages:
`Numpy`, `SciPy` and `Matplotlib`. It draws some ideas from `scikit-learn` and
`scikit-image`, and may in future use these two more. The package also keeps a
copy of  `tiffile.py` by Christoph Gohlke (version 2013.01.18) to load
multi-frame TIFF files.

The script `frame_viewer.py`, a simple GUI wrapper for `imfun`, also uses
`Traits` and `TraitsUI`.

## License

Except for files, adopted from external sources (such as `tiffile.py`) the code
is GPL. Other open licensing (e.g. MIT LGPL) can be leased on demand.

## Publications
The software has been used in production of the following journal articles:
  - PMID: 23219568
  - PMID: 23211964
  - PMID: 24218625
  - PMID: 24692513 
