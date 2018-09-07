# Image-funcut

In 2017–2018 supported by RSF grant 17-74-20089

## Overview

The `image-funcut` project is kind of a sandbox or testbed for utilities to
view, analyse and transform two-photon microscopy data or any other series of
images.

At the moment, the project includes the following `Python` modules:
  - `imfun.bwmorph`: helper functions for black-white morphology and binary
    masks 
  - `imfun.cluster`: naive implementations of a few clustering algorithms and
      distance functions
    - `imfun.cluster.som`: implementation of Self-organizing Kohonen maps clustering
      algorithm
  - `imfun.components`: Component analysis an factorization
    - `imfun.components.pca`:  PCA
    - `imfun.components.ica`: ICA 
  - `imfun.core`: miscellaneous helper functions
    - `imfun.core.fnutils`: a few functional programming-inspired utils
  - `imfun.external`: External modules 
  - `imfun.filt`: various filtering utils
  - `imfun.fnmap`: collection of functions which project XYT data to 2D images
    in various non-trivial ways
  - `imfun.fseq`: a keystone module: Class definitions and functions to read
    from files of several formats and represent sequences of images (both lazy
    and not) and operations on them.
  - `imfun.multiscale`: Multiscale operations: image pyramids and the like.
    - `imfun.multiscale.atrous`: comprises functions for à trous wavelet transform and
    related utilities. (Synonims: starlet transform, stationary wavelet
    transform, non-decimated 
    wavelet transform). Besides transform, there are utility functions to smooth
    input data with B-splines, remove trends in data or enhance data by noise
    suppression. 
    - `imfun.multiscale.emd`: a stub for empirical mode decomposition functions
    - `imfun.multiscale.mmt`: multiscale median transform and hybrid median/starlet
      transform 
    - `imfun.multiscale.mvm`: an implementation of the Multiscale Vision Model
  - `imfun.ofreg`: Optical flow and image registration 
  - `imfun.opt`: a stub for optimization functions
  - `imfun.synthdata`: functions to create simple synthetic data sets should be
    collected here
  - `imfun.track`: functions to track objects in a changing environment or
    align frames should be collected here
  - `imfun.ui` : User interface related modules
    - `imfun.ui.Picker`: Picker class -- a matplotlib-based backend-independent user
      interface to operate on `fseq` instances
  - `imfun.io`: Reading various file formats
    - `imfun.io.leica`: parsing XML files produced by Leica Software during export
    - `imfun.io.mes`: reading MES files, as created by a Femtonics microscope
    - `imfun.io.MLFImage`: interface to load MLF files produced by Moor laser
      speckle imaging device.
  - `scripts/frame_viewer.py`: a Traits-based GUI wrapper over `imfun.ui` and other
    modules  with additional features. Currently non-operational

One of the motivations to start this project was a *func*tional
programming approach to image data analysis, hence the name. Also, it's like a
final-cut, but with some (geeky) fun.

## Installation
### Linux and Mac, with Anaconda

First, install dependencies:
```bash
conda install pandas numba matplotlib h5py
conda install scipy scikit-image pygments 
conda install traits traitsui
conda install -c conda-forge pathos
conda install -c conda-forge opencv

#pip install https://github.com/pyimreg/imreg/archive/master.zip
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
