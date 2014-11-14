### For speckle imaging data
### MLF is for Moor Instruments data

### Let's decide I used hexdump on a mlf file and figured out the format

import numpy as np
import numbers


# table for v2.0 files
# format: label: (address, numElements, typeOfElement)
mlfdescr_v2 = {
    'date':      (0x1800, 1, '|S10'),
    'time':      (0x1c00, 1, 'S9'),
    'nframes':   (0x8000, 1, np.uint32),
    'xdim' :     (0x8008, 1, np.uint32),
    'ydim':      (0x8010, 1, np.uint32),
    'dt':        (0x8018, 1, np.uint32), # in ms
    'exposure' : (0x8028, 1, np.uint32),
    'sfilter' :  (0x8038, 1, np.uint32),
    'mode':      (0x8048, 1, np.uint32),
    'tfilter':   (0x8050, 1, np.uint32),
    'data_start':(0x10000, None, np.uint16)
    }

# table for v3.0 files
# format: label: (address, numElements, typeOfElement)
mlfdescr_v3 = {
    'version':   (0x0100, 1, np.int64),
    'nsections': (0x108, 1, np.int64),
    'date':      (0x0900, 1, '|S10'),
    'time':      (0x0980, 1, '|S9'),
    'nframes':   (0x1000, 1,np.int64),
    'xdim':      (0x1008, 1, np.int64),
    'ydim':      (0x1010, 1, np.int64),
    'dt' :       (0x1018, 1, np.int64), # in us
    'exposure':  (0x1028, 1, np.int64),
    'gain':      (0x1030, 1, np.int64),
    'sfilter':   (0x1038, 1, np.int64),
    'mode':      (0x1048, 1, np.int64),
    'tfilter':   (0x1050, 1, np.int64),
    'data_start':(0x10000, None, np.uint16),
    }

def read_at(fid, pos, Nelem=1, dtype=np.uint16, seek_opt=0):
    fid.seek(pos, seek_opt)
    return np.fromfile(fid, dtype, Nelem)

# this needed to guess file type from file header
known_headers = {
    "Moor FLPI Live Image Data file V2.0":mlfdescr_v2,
    "moorFLPI Live Image Data File V3.0": mlfdescr_v3
    }

def read_header(name):
    """Read file header and return the correct address table
    """
    with open(name, 'rb') as fid:
        magic = fid.read(64)
        tables = [(k,v) for k,v in known_headers.items() if k in magic]
        fid.seek(0)
        if len(tables):
            return tables[0]
        else:
            print "Unknown MLF file format"

class MLF_Image:
    def __init__(self, fname):
        self.fname = fname
        magic,table =  read_header(fname)
        if table is None:
            print "Can't load data, quitting the class"
            return
        self.fid = open(fname, 'rb')
        meta = {k:read_at(self.fid, *v)[0] for k,v in table.items()
                     if not None in v}
        self.data_start = table['data_start'][0]
        self.__dict__.update(meta)
        if table == mlfdescr_v3:
            self.dt/=1000.
	self.shape = (self.ydim, self.xdim)
        self.dim = self.xdim*self.ydim
        
    def read_value(self, pos, dtype=np.uint32,seek_opt=0):
        return read_at(self.fid, pos, 1, dtype=dtype,seek_opt=seek_opt)[0]

    def location2index(self, loc):
	#nrows,ncols = self.ydim,self.xdim
	r,c=loc
	return 2*(r*self.xdim + c)
	
    def read_timeslice(self, loc):
	index= self.location2index(loc)
	positions = 4*np.arange(self.nframes)*self.dim + index
	positions += self.data_start
	values = [self.read_value(p,dtype=np.uint16)  for p in positions]
	return np.array(values)


    def read_next_frame(self, pos=0, seek_opt=1):
        arr = read_at(self.fid, pos, self.dim, seek_opt=seek_opt)
        return arr.reshape((self.ydim,self.xdim))

    def read_frame(self, n):
	n = n%self.nframes
	pos = n*self.dim*4 + self.data_start
	frame = read_at(self.fid, pos, self.dim)
	return frame.reshape(self.shape)

    def __getitem__(self, val):
	indices = range(self.nframes)
        try:
            iter(indices[val])
        except TypeError:
            #print "MLF_Image: indices[val] doesn't support iteration?"
            #print val, type(val), indices
            if isinstance(val, numbers.Number):
                return self.read_frame(val)
            print """MLF_Image: indices[val] doesn't support iteration and is not
	a number"""
            print val, type(val), indices
            return self.read_frame(0)
        else:
            return map(self.read_frame, indices[val])

    def flux_frame_iter(self):
        frame_count = 0
        self.fid.seek(self.data_start)
        shift = self.xdim*self.ydim
        while frame_count < self.nframes:
        #while True:
            try:
                flux_frame = self.read_next_frame() # get "flux frame"
		self.fid.seek(shift*2, 1) # omit "dc frame"
                frame_count += 1
                yield flux_frame
            except:
                break

    def frame_iter(self):
        frame_count = 0
        self.fid.seek(self.data_start)
        while frame_count < self.nframes:
            flux_frame = self.read_next_frame() # seek from current
            dc_frame = self.read_next_frame() # seek from current
            frame_count += 1
            yield flux_frame, dc_frame
