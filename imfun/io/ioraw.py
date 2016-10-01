### Support for some raw binary data formats (MLF, PLS, PIC)
### MLF is format by Moor Instruments
### PLS is format by D. Postnov (jr)
### PIC format is used on CCDB

import numpy as np
from numpy import uint16, uint32, uint64
from numpy import int16, int32, int64
import numbers

def read_at(fid, pos, Nelem=1, dtype=uint16, seek_opt=0):
    fid.seek(pos, seek_opt)
    return np.fromfile(fid, dtype, Nelem)


# table for MLF v2.0 files
# format: label: (address, numElements, typeOfElement)
_mlfdescr_v2 = {
    'date':      (0x1800, 1, '|S10'),
    'time':      (0x1c00, 1, 'S9'),
    'nframes':   (0x8000, 1, uint32),
    'ncols' :     (0x8008, 1, uint32),
    'nrows':      (0x8010, 1, uint32),
    'dt':        (0x8018, 1, uint32), # in ms
    'exposure' : (0x8028, 1, uint32),
    'sfilter' :  (0x8038, 1, uint32),
    'mode':      (0x8048, 1, uint32),
    'tfilter':   (0x8050, 1, uint32),
    'data_start':(0x10000, None, uint16),
    }

# table for MLF v3.0 files
# format: label: (address, numElements, typeOfElement)
_mlfdescr_v3 = {
    'version':   (0x0100, 1, np.int64),
    'nsections': (0x108, 1, np.int64),
    'date':      (0x0900, 1, '|S10'),
    'time':      (0x0980, 1, '|S9'),
    'nframes':   (0x1000, 1,np.int64),
    'ncols':      (0x1008, 1, np.int64),
    'nrows':      (0x1010, 1, np.int64),
    'dt' :       (0x1018, 1, np.int64), # in us
    'exposure':  (0x1028, 1, np.int64),
    'gain':      (0x1030, 1, np.int64),
    'sfilter':   (0x1038, 1, np.int64),
    'mode':      (0x1048, 1, np.int64),
    'tfilter':   (0x1050, 1, np.int64),
    'data_start':(0x10000, None, np.uint16),
    }

_PLSI_fields = {
    'ncols': (64, 1, uint64),
    'nrows': (72, 1, uint64),
    'nframes': (80, 1, uint64),
    'dt': (88, 1, uint64),      # in ms
    'exposure': (96, 1, uint64), # in us
    'data_start':(30*1024, None, uint16),
    }



def read_at(fid, pos, Nelem=1, dtype=np.uint16, seek_opt=0):
    fid.seek(pos, seek_opt)
    return np.fromfile(fid, dtype, Nelem)

# this needed to guess file type from file header
_known_headers = {
    "Moor FLPI Live Image Data file V2.0":_mlfdescr_v2,
    "moorFLPI Live Image Data File V3.0": _mlfdescr_v3,
    "LSI Recorder processed file v1.0 type: tLASCA 25":_PLSI_fields
    }



def read_header(name):
    """Read file header and return the correct address table
    """
    with open(name, 'rb') as fid:
        magic = fid.read(64).strip().lower()
        tables = [(k,v) for k,v in _known_headers.items() if k.strip().lower() in magic]
        fid.seek(0)
        if len(tables):
            return tables[0]
        else:
            print "Unknown MLF/PLS file format"

class PLSI:
    def __init__(self, fname):
        self.fname = fname
        magic,table =  read_header(fname)
        if table is None:
            raise InputError("Can't load data, unrecognized file format")

        self.fid = open(fname, 'rb')
        meta = {k:read_at(self.fid, *v)[0] for k,v in table.items() if not None in v}
        self.__dict__.update(meta)


        if table in (_mlfdescr_v2, _mlfdescr_v3):
            # MLF formats have 'flow' and 'contrast' frames
            self.have_double_frames = True
            self.have_tstamps = False
            self.order = 'C'
            if table == _mlfdescr_v3:
                self.dt/=1000.
        if table == _PLSI_fields:
            # PLS format has a timestamp field before each frame
            self.have_tstamps = True
            self.have_double_frames = False
            self.order = 'F'
            

        self.data_start = table['data_start'][0]
        self.dtype = table['data_start'][2]
                    
	self.shape = (self.nrows, self.ncols)
        self.npix = np.prod(self.shape)

        self.vbyte_size = np.nbytes[self.dtype]
        self.tstamp_dtype = uint64
        self.frame_byte_size = self.npix*self.vbyte_size

        if self.have_tstamps:
            self.frame_byte_size += np.nbytes[self.tstamp_dtype]
        if self.have_double_frames:
            self.frame_byte_size *= 2 

        
    def read_value(self, pos, dtype=np.uint16,seek_opt=0):
        return read_at(self.fid, pos, 1, dtype=dtype,seek_opt=seek_opt)[0]

    def location2index(self, loc):
	#nrows,ncols = self.ydim,self.xdim
	r,c=loc
	return self.vbyte_size*(r*self.ncols + c)
	
    def read_timeslice(self, loc):
	index= self.location2index(loc)
	positions = self.data_start + self.frame_byte_size + index
	values = [self.read_value(p,dtype=self.dtype)  for p in positions]
	return np.array(values)

    def read_next_frame(self, pos=0):
        if self.have_tstamps:
            pos += np.nbytes[self.tstamp_dtype]
        arr = read_at(self.fid, pos, self.npix, seek_opt=1)
        if self.have_double_frames:
            self.fid.seek(self.frame_byte_size/2,1)
        return arr.reshape((self.nrows,self.ncols),order=self.order)

    def read_frame(self, n):
	n = n%self.nframes
        pos = int(self.data_start + n*self.frame_byte_size)
        if self.have_tstamps:
            tstamp = read_at(self.fid, pos, 1, self.tstamp_dtype)
            frame = read_at(self.fid, 0, self.npix, self.dtype, seek_opt=1)
        else:
            frame = read_at(self.fid, pos, self.npix, self.dtype)
	return np.reshape(frame, self.shape, order=self.order) # note order may be different for MLF!

    def __getitem__(self, val):
	indices = range(self.nframes)
        try:
            iter(indices[val])
        except TypeError:
            #print "MLF_Image: indices[val] doesn't support iteration?"
            #print val, type(val), indices
            if isinstance(val, numbers.Number):
                return self.read_frame(val)
            print """PLS: indices[val] doesn't support iteration and is not a number"""
            print val, type(val), indices
            return self.read_frame(0)
        else:
            return np.array(map(self.read_frame, indices[val]))

    def frame_iter(self):
        frame_count = 0
        self.fid.seek(self.data_start)
        while frame_count < self.nframes:
            try:
                flux_frame = self.read_next_frame() # get "flux frame"
                #if self.have_double_frames:
                #    self.fid.seek(shift*2, 1) # omit "dc frame"
                frame_count += 1
                yield flux_frame
            except:
                break

    def __double_frame_iter(self):
        frame_count = 0
        self.fid.seek(self.data_start)
        while frame_count < self.nframes:
            flux_frame = self.read_next_frame() # seek from current
            dc_frame = self.read_next_frame() # seek from current
            frame_count += 1
            yield flux_frame, dc_frame



class PICImage(object):
    def read_pic(name):
        fid = open(name, 'rb')
        nx,ny,nz = map(int, np.fromfile(fid, uint16, 3))
        start_frames = 0x4c
        fid.seek(start_frames,0)
        frames = np.fromfile(fid, uint8, nx*ny*nz).reshape(nz,nx,ny)
        
        meta_start = start_frames + nx*ny*nz + 0x10
        meta = load_meta(fid, meta_start)
        return frames, meta

    def load_meta(fid, meta_start, nread=38):
        acc = []
        step = 0x60
        fid.seek(meta_start,0)
        for k in range(0,nread):
            entry = fid.read(0x30).strip('\x00')
            acc.append(entry)
            fid.seek(0x30,1)
        return acc

    def get_axes(meta):
        ax_info = [e for e in meta if 'axis' in e.lower() and 'microns' in e.lower()]
        acc = []
        for ax in ax_info:
            x = ax.split()[-2:]
            acc.append((float(x[0]), x[1]))
        
        return acc
