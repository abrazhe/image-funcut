### Support for speckle imaging data for D. Postnov's software

import numpy as np
from numpy import uint64, uint16

_description_ = """
1) 0-63,  64bytes - header stating what type of meta data it is. Let's say i will put "LSI Recorder processed file v1.0 type: tLASCA 25" - chars sequence
2) 64-71,  8bytes - sizeX of the single frame (Uint64)
3) 72-79,  8bytes - sizeY of the single frame (Uint64)
4) 80-87,  8bytes - sizeT - number of frames (Uint64)
5) 88-95,  8bytes - sampling - time between two frames in milliseconds (Uint64)
6) 96-103,  8bytes - exposure time in microseconds (Uint64)
6) 104-111, 8bytes - used comments number (Uint64)
7) 104 - 30720  - area for comments of fixed size, namely timestamp (8 bytes Uint64) + text (140bytes ), around 200 comments are possible
8) 30720 (30*1024) - start of frames with timestamps, each frame encoded:
8 bytes for timestamp (Uint64)
then sizeX*sizeY*Uint16 elements. X will be columns and Y will be rows.
"""


def read_at(fid, pos, Nelem=1, dtype=np.uint64, seek_opt=0):
    fid.seek(pos, seek_opt)
    return np.fromfile(fid, dtype, Nelem)

## def read_value(self, pos, dtype=np.uint32,seek_opt=0):
##         return read_at(self.fid, pos, 1, dtype=dtype,seek_opt=seek_opt)[0]

_file_header = "LSI Recorder processed file v1.0 type: tLASCA 25"


_fields = {
    'ncols': (64, 1, uint64),
    'nrows': (72, 1, uint64),
    'nframes': (80, 1, uint64),
    'dt': (88, 1, uint64),      # in ms
    'exposure': (96, 1, uint64), # in us
    'data_start':(3*1024, None, uint16)
    }


def read_header(name):
    """Read and return file header
    """
    with open(name, 'rb') as fid:
        magic = fid.read(64)
        if magic.strip().lower() == file_header.strip().lower():
            return magic,_fields

class PLSImages(object):
    def __init__(self, fname):
        self.fname = fname
        magic,table = read_header(fname)
        if table is None:
            raise InputError("Can't load data, unrecognized file format")
        self.fid = open(fname, 'rb')
        meta = {k:read_at(self.fid, *v)[0] for k,v in table.items() if not None in v}
        self.__dict__.update(meta)
        self.data_start = table['data_start'][0]
        self.dtype = table['data_start'][2]
        self.shape = (self.nrows,self.ncols)
        self.npix = np.prod(self.shape)
        self.frame_byte_size = np.nbytes[uint64] + np.nbytes[self.dtype]*self.npix
    def read_frame(self, n):
        n = n%self.nframes
        pos = self.data_start + n*self.frame_byte_size
        pos = int(pos)
        print pos
        tstamp = read_at(self.fid, pos, 1, uint64)
        frame = read_at(self.fid, 0, self.npix, self.dtype, seek_opt=1)
        return np.reshape(frame, self.shape, order='F'), tstamp

                      
            

