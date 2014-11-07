# Routines to read MES files

from __future__ import division

import itertools as itt
import numpy as np
from scipy import io

import lib

def load_meta(file_name,max_records=1000):
    "Load meta information from a MES file"
    var_names = ["Df%04d"%k for k in range(1,max_records+1)]
    meta = io.loadmat(file_name, variable_names=var_names, appendmat=False)
    return meta

## def describe_file(name):
##     meta = load_meta(name)
##     out_str = "\n# File: %s\n"%name
##     out_str += '-'*(len(name)+10) + '\n'
##     keys = record_keys(meta)
##     out_str +=  "## Data structures:\n"
##     out_str += ', '.join(keys) + '\n'
##     for key in keys:
##         out_str += describe_record(key, meta)
##     return out_str

## def describe_record(key, meta):
##         out_str = '### ' +  key + '\n'
##         timestamps = [x[0][0] for x in meta[key]['MeasurementDate']
##                       if np.prod(x[0].shape)>0 ]
##         image_names = [map(get_field, (x['ImageName'], x['Context'], x['Channel'])) for x in meta[key]]
##         timestamps = sorted(timestamps)
##         context = [c[0][0] for c in meta[key]["Context"]]
##         #print 'Context types:', set(context)
##         out_str += '  + Total context entries: %d\n'%len(context)
##         out_str += '  + Number of context entries of each type: ' + \
##                    str([(x,sum(np.array(context)==x)) for x in set(context)])
##         out_str += '\n'
##         out_str += '  + Timestamps (first,last): '
##         out_str += timestamps[0] + ', ' +\
##                    (len(timestamps)>1 and timestamps[-1] or '') + '\n'
##         out_str += '  + Image names: \n' 
##         for x in image_names:
##             out_str += "    - " + str(tuple(x)) + "\n"
##         return out_str
    

## TODO use [()] together with squeeze_me argument for scipy.io.loadmat

#def record_keys(meta):
#    return sorted(filter(lambda x: 'Df' in x, meta.keys()))
    
def only_measures(entry):
    return entry[entry['Context']=='Measure']

def first_measure(entry):
    measures = only_measures(entry)
    if len(measures):
        return measures[0]

def check_record_type(record, file_name):
    if type(record) is int:
        record = 'Df%04d'%record
    elif type(record) is str:
        if not ('Df' in record):
            record = 'Df%04d'%int(record)
    if type(record) is str:
        meta = load_meta(file_name)
        record = meta[record]
    return record

class MAT_Record:
    """Base functions to read entries from MES-related MATv7 files"""
    def __init__(self, file_name, recordName, ch=None):
        self.ch = ch
        self.record = check_record_type(recordName, file_name)
        self.contexts = self.record["Context"]
        self.file_name = file_name
        self.dx = self.get_field(self.record['WidthStep'])
        self.timestamps = [x[0][0] for x in self.record['MeasurementDate']
                           if np.prod(x[0].shape)>0 ]
        self.img_names = map(self.get_field, self.record['ImageName'])
        self.channels = [self.get_field(x).lower() for x in self.record['Channel']]


    def _get_stream(self, name):
        rec = io.loadmat(self.file_name,variable_names=[name],appendmat=False)
        return rec[name]    
    def _get_recs(self, names):
        return io.loadmat(self.file_name, variable_names=names,appendmat=False)
    def is_zstack(self):
        "Check if an entry is a z-stack"
        entry = self.record
        return 'Zstack' in entry['Context'] and 'Zlevel' in entry.dtype.names
    def is_supported(self):
        #entry = self.record
        return self.is_zstack() or self.is_timelapse()
    def is_timelapse(entry):
        """Check if an entry is an XYT measurement. It must have a 'FoldedFrameInfo' propery"""
        return 'FoldedFrameInfo' in entry.dtype.names
    def get_field(self, entry):
        return entry[0][0]

class H5_Record:
    "Base functions to read entries from MES-related HDF5 MAT files"
    def __init__(self, file_name, recordName,ch=None):
        self.ch = ch
        self.h5file = h5py.File(file_name)
        tkeyf = self._get_str_field
        nkeyf = self._get_num_field
        self.record = recordName
        self.img_names = np.array(tkeyf('ImageName'))
        self.contexts = np.array(tkeyf('Context'))
        self.channels = tkeyf('Channel')
        self.nchannels = len(np.unique(self.channels))
        self.timestamps = [s for s in tkeyf('MeasurementDate')
                           if s!='\x00\x00']
        self.dims = nkeyf('DIMS')
        self.dx = nkeyf('WidthStep')[0]


    def _get_recs(self, names):
        return {n:self.h5file[n] for n in names}
    def _get_num_field(self, rec):
        return read_numentry_h5(self.h5file, '/'.join((self.record,rec)))
    def _get_str_field(self, rec):
        return read_txtentry_h5(self.h5file, '/'.join((self.record,rec)))


class MES_Record:
    pass

class ZStack(MES_Record):
    kind = 'Zstack'
    def load_data(self, ch=None):
        if ch is None: ch = self.ch
        #outmeta = {'ch':ch}
        outmeta = dict(ch=ch, timestamp=self.timestamps[0])
        outmeta['axes'] = lib.alist_to_scale([(self.dz, 'um'),
                                              (self.dx, 'um')])
        var_names = self.img_names

        nchannels = self.nchannels
        if ch is None or ch == 'all':
            ## TODO: use smth like np.concatenate(map(lambda m:\
            ## m.reshape(m.shape+(1,)), (x1,x2,x3)), 3)
            sh = tuple(self.base_shape) + (max(3, nchannels),)
            print 'Shape', sh, self.base_shape, nchannels
            recs = self._get_recs(var_names)
            stream = lib.clip_and_rescale(np.array([recs[n] for n in var_names]))
            data = np.zeros(sh)
            framecount = 0
            print 'Shape2:', stream[0].shape
            for k in xrange(0, self.nframes//nchannels,nchannels):
                for j in xrange(nchannels):
                    data[framecount,...,j] = stream[k+j]
                framecount += 1
                    
        else:
            if type(ch) is int :
                var_names = var_names[ch::nchannels]
            elif type(ch) is str :
                var_names = [n for n,c in zip(var_names,self.channels) if ch.lower() in c]
            recs = self._get_recs(var_names)
            data = np.array([recs[n] for n in var_names])
        return data, outmeta

class ZStack_mat(ZStack, MAT_Record):
    def __init__(self, file_name, recordName, ch=None):
        MAT_Record.__init__(self, file_name, recordName, ch)
        self.dz = self._get_zstep()
        self.nframes = len(self.record)
        self.frame_shape = tuple(self.get_field(self.record[0]['DIMS']))
        self.nchannels = len(np.unique(self.channels))
        self.base_shape = (self.nframes//self.nchannels, ) + self.frame_shape
    def _get_zstep(self):
        ch1 = self.get_field(self.record[0]['Channel'])
        levels = self.record[self.record['Channel']==ch1]['Zlevel']
        return np.mean(np.diff(map(self.get_field, levels)))


class ZStack_h5(ZStack, H5_Record):
    def __init__(self,file_name, recordName, ch=None):
        H5_Record.__init__(self, file_name, recordName, ch)
        self.dz = self._get_zstep()
        self.nframes = len(self.img_names)
        self.frame_shape = self.dims[0]
        self.base_shape = (self.nframes//self.nchannels, ) + tuple(self.frame_shape)

    def _get_zstep(self):
        ch1 = self.channels[0]
        levels = self._get_num_field('Zlevel')
        return np.mean(np.diff(levels[self.channels==ch1]))

class Timelapse(MES_Record):
    kind='Timelapse'
    def load_data(self,ch=None):
        if ch == None: ch = self.ch
        outmeta = dict(ch=ch, timestamp=self.timestamps[0])
        outmeta['axes'] = lib.alist_to_scale([(self.dt,'s'), (self.dx, 'um')])
        nframes, nlines, line_len = self.nframes, self.nlines, self.line_length
        base_shape = (nframes-1, nlines, line_len)
        streams = self._load_streams(ch=ch)
        if len(streams) == 1:
            data = np.zeros(base_shape, dtype=streams[0].dtype)
            for k,f in enumerate(self._reshape_frames(streams[0])):
                data[k] = f
        else:
            streams = [lib.clip_and_rescale(s) for s in streams]
            reshape_iter = itt.izip(*map(self._reshape_frames, streams))
            sh = base_shape + (max(3, len(streams)),)
            data = np.zeros(sh, dtype=streams[0].dtype)
            for k, a in enumerate(reshape_iter):
                for j, f in enumerate(a):
                    data[k,...,j] = f
        self.data = data
        self.outmeta = outmeta
        return  data, outmeta
    def _load_streams(self,ch=None):
        if ch is None: ch = self.ch
        var_names = self.img_names
        if not (ch is None or ch=='all'):
            if isinstance(ch, int):
                var_names = [var_names[ch]]
            elif isinstance(ch, str):
                var_names = [n for n,c
                             in zip(var_names, self.channels)
                             if ch.lower() in c.lower()]
        streams = map(self._get_stream, var_names)
        if len(streams)==0:
            raise IndexError("MES.Timelapse: can't load record%s"%self.recordName)
        return streams

class  Timelapse_mat(Timelapse, MAT_Record):
    def __init__(self, file_name, recordName, ch=None):
        MAT_Record.__init__(self, file_name, recordName, ch)
        self.measures = only_measures(self.record)
        self.dt = self.get_sampling_interval()
        nframes, (line_len, nlines) = self.get_xyt_shape()
        self.img_names = [x[0] for x in self.measures['ImageName']]
        self.channels = [x[0] for x in self.measures['Channel']]
    def _reshape_frames(self, stream):
        nlines,nframes = map(int, (self.nlines, self.nframes))
        return (stream[:,k*nlines:(k+1)*nlines].T for k in xrange(1,nframes))
    def get_xyt_shape(self,):
        'return (numFrames, (side1,side2))'
        m = first_measure(self.record)
        self.nlines = int(m['FoldedFrameInfo']['numFrameLines'][0])
        self.nframes = int(m['FoldedFrameInfo']['numFrames'][0])
        self.line_length = m['DIMS'][0][0]
        return self.nframes, (self.line_length, self.nlines)
    def get_sampling_interval(self,ffi):
        ffi = self.get_ffi(self.record)
        nframes = long(ffi['numFrames'])
        tstart = float(ffi['firstFrameStartTime'])
        tstop = float(ffi['frameTimeLength'])
        return (tstop-tstart)/nframes
    def get_ffi(self): 
        return first_measure(self.record)['FoldedFrameInfo'][0]

class Timelapse_h5(Timelapse, H5_Record):
    def __init__(self, file_name, recordName, ch=None):
        H5_Record.__init__(self, file_name, recordName, ch)
        self.img_names = self.img_names[[self.contexts=='Measure']]
        self.line_length = self.dims[0,0]
        ffi = get_ffi_h5(self.h5file, recordName)
        self.__dict__.update(ffi)
        self.frame_2d_shape = (self.line_length, self.nlines)
        self.dt = (self.tstop-self.tstart)/self.nframes
    def _reshape_frames(self, stream):
        nlines,nframes = map(int, (self.nlines, self.nframes))
        return (stream[k*nlines:(k+1)*nlines,:] for k in xrange(1,nframes))
    def _get_stream(self, name):
        return self.h5file[name]
    
# --- Routines for HDF5 Matlab files (to be refactored later) ----------

import h5py

def field_to_string_h5(h5file, objref):
    return ''.join(unichr(c) for c in h5file[objref])

def read_txtentry_h5(h5file, refstr):
    refs = h5file[refstr]
    return np.array([field_to_string_h5(h5file, e) for e in refs[0,:]])

def read_numentry_h5(h5file, refstr):
    refs = h5file[refstr]
    return np.squeeze(np.array([h5file[e] for e in refs[0,:]]))

def get_ffi_h5(h5file, record):
    ref =  h5file['/'+record+'/FoldedFrameInfo']
    group = h5file[ref[0,0]]
    keyf = lambda s: np.squeeze(np.array(group[s]))[()]
    names = [('nframes', 'numFrames'),
             ('tstart', 'firstFrameStartTime'),
             ('tstop', 'frameTimeLength'),
             ('nlines', 'numFrameLines'),
             ('transverseStep', 'TransverseStep'),
             ('firstFramePos', 'firstFramePos')]
    return {n[0]:keyf(n[1]) for n in names}

## TODO: make class with __repr__ instead
def describe_file_h5(file_name):
    with h5py.File(file_name) as f: 
        record_keys = [k for k in f.keys() if 'Df' in k]
        for key in record_keys:
            print key, ':', describe_record_h5(f,key, f[key])

def describe_record_h5(h5file, key, h5group):
    contexts = read_txtentry_h5(h5file, key+'/Context')
    context_unique = np.unique(contexts)
    return context_unique
