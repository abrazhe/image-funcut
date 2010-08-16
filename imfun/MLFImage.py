### For speckle imaging data
### MLF is for Moor Instruments data

### Let's decide I used hexdump on a mlf file and figured out the formatq

import numpy as np

mlfdescr = {
    'date': 0x1800,
    'time': 0x1c00,
    'nframes': 0x8000,
    'xdim' : 0x8008,
    'ydim': 0x8010,
    'dt': 0x8018,
    'exposure' : 0x8028,
    'sfilter' : 0x8038,
    'mode': 0x8048,
    'tfilter': 0x8050,
    'data_start': 0x10000
    }

def read_at(fid, pos, Nelem, dtype=np.uint16, seek_opt=0):
    fid.seek(pos, seek_opt)
    return np.fromfile(fid, dtype, Nelem)

class MLF_Image:
    def __init__(self, fname):
        self.fname = fname
        self.fid = open(fname, 'rb')
        self.xdim  = self.read_value(mlfdescr['xdim'])
        self.ydim  = self.read_value(mlfdescr['ydim'])
        self.dt = self.read_value(mlfdescr['dt'])
        self.nframes = self.read_value(mlfdescr['nframes'])
        self.exposure = self.read_value(mlfdescr['nframes'])
        self.sfilter = self.read_value(mlfdescr['sfilter'])
        self.mode = self.read_value(mlfdescr['mode'])
        self.tfilter = self.read_value(mlfdescr['tfilter'])
        self.dim = self.xdim*self.ydim
        
    def read_value(self, pos):
        return long(read_at(self.fid, pos, 1, np.short))

    def read_frame(self, pos=0, seek_opt=1):
        arr = read_at(self.fid, pos, self.dim, seek_opt=seek_opt)
        return arr.reshape((self.ydim,self.xdim))

    def flux_frame_iter(self):
        frame_count = 0
        self.fid.seek(mlfdescr['data_start'])
        shift = self.xdim*self.ydim
        while frame_count < self.nframes:
            flux_frame = self.read_frame() # seek from current
            dc_frame = self.read_frame() # seek from current
            frame_count += 1
            yield flux_frame


    def frame_iter(self):
        frame_count = 0
        self.fid.seek(mlfdescr['data_start'])
        shift = self.xdim*self.ydim
        while frame_count < self.nframes:
            flux_frame = self.read_frame() # seek from current
            dc_frame = self.read_frame() # seek from current
            frame_count += 1
            yield flux_frame, dc_frame
