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

def describe_file(name):
    meta = load_meta(name)
    out_str = "\n# File: %s\n"%name
    out_str += '-'*(len(name)+10) + '\n'
    keys = record_keys(meta)
    out_str +=  "## Data structures:\n"
    out_str += ', '.join(keys) + '\n'
    for key in keys:
        out_str += describe_record(key, meta)
    return out_str

def describe_record(key, meta):
        out_str = '### ' +  key + '\n'
        timestamps = [x[0][0] for x in meta[key]['MeasurementDate']
                      if np.prod(x[0].shape)>0 ]
        image_names = [map(get_field, (x['ImageName'], x['Context'], x['Channel'])) for x in meta[key]]
        timestamps = sorted(timestamps)
        context = [c[0][0] for c in meta[key]["Context"]]
        #print 'Context types:', set(context)
        out_str += '  + Total context entries: %d\n'%len(context)
        out_str += '  + Number of context entries of each type: ' + \
                   str([(x,sum(np.array(context)==x)) for x in set(context)])
        out_str += '\n'
        out_str += '  + Timestamps (first,last): '
        out_str += timestamps[0] + ', ' +\
                   (len(timestamps)>1 and timestamps[-1] or '') + '\n'
        out_str += '  + Image names: \n' 
        for x in image_names:
            out_str += "    - " + str(tuple(x)) + "\n"
        return out_str
    

#def get_timestamps(entry):
#    timestamps = [get_field(x) for x in entry['MeasurementDate']
#                  if np.prod(x[0].shape)>0 ]
#    return timestamps
    

## def _get_timestamp(entry, context='Measure'):
##     elist =  entry[entry['Context']==context]['MeasurementDate']
##     filtered = [str(e[0]) for e in elist if len(e[0])]
##     if len(filtered) == 1:
##         return filtered[0]
##     else: return filtered


## TODO use [()] together with squeeze_me argument for scipy.io.loadmat
def get_field(entry):
    return entry[0][0]


def record_keys(meta):
    return sorted(filter(lambda x: 'Df' in x, meta.keys()))

def is_supported(entry):
    return is_zstack(entry) or is_timelapse(entry)
    
def is_zstack(entry):
    "Check if an entry is a z-stack"
    return 'Zstack' in entry['Context'] and 'Zlevel' in entry.dtype.names

def is_timelapse(entry):
    """Check if an entry is an XYT measurement.
    It must have a 'FoldedFrameInfo' propery"""
    return 'FoldedFrameInfo' in entry.dtype.names

def only_measures(entry):
    return entry[entry['Context']=='Measure']

def first_measure(entry):
    measures = only_measures(entry)
    if len(measures):
        return measures[0]

def get_ffi(record):
    return first_measure(record)['FoldedFrameInfo'][0]

def get_sampling_interval(ffi):
    nframes = long(ffi['numFrames'])
    tstart = float(ffi['firstFrameStartTime'])
    tstop = float(ffi['frameTimeLength'])
    return (tstop-tstart)/nframes

def get_xyt_shape(entry):
    'return numFrames, (side1,side2)'
    m = first_measure(entry)
    nlines = int(m['FoldedFrameInfo']['numFrameLines'][0])
    nframes = int(m['FoldedFrameInfo']['numFrames'][0])
    line_length = m['DIMS'][0][0]
    return nframes, (line_length, nlines)

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

    
def load_timelapse_v7(file_name, record, ch=None):
    record = check_record_type(record, file_name)
    outmeta = {'ch':ch}

    measures = only_measures(record)

    ffi = get_ffi(record)
    dt = get_sampling_interval(ffi)
    dx = get_field(measures[0]['WidthStep'])
    outmeta['axes'] = lib.alist_to_scale([(dt,'s'), (dx, 'um')])
    
    outmeta['timestamp'] = measures['MeasurementDate'][0][0]
    nframes, (line_len, nlines) = get_xyt_shape(record)
    def _reshape_frames(stream):
        return (stream[:,k*nlines:(k+1)*nlines].T for k in xrange(1,nframes))
    def _load_streams():
        var_names = [x[0] for x in measures['ImageName']]
        channels = [x[0] for x in measures['Channel']]
        if not (ch is None or ch == 'all'):
            if type(ch) is int :
                var_names = [var_names[ch]]
            elif type(ch) is str: # can be one of 'r', 'g', 'b'
                var_names = [n for n,c in zip(var_names,channels) if ch.lower() in c.lower()]
        recs = io.loadmat(file_name,variable_names=var_names,appendmat=False)
        streams = [recs[n] for n in var_names if n in recs]
        if len(streams)==0:
            raise IndexError("can't load record")
        return streams

    streams = _load_streams()
    base_shape = (nframes-1, nlines, line_len)
    if len(streams) == 1:
        data = np.zeros(base_shape, dtype=streams[0].dtype)
        for k,f in enumerate(_reshape_frames(streams[0])):
            data[k] = f
    else:
        streams = [lib.clip_and_rescale(s) for s in streams]
        reshape_iter = itt.izip(*map(_reshape_frames, streams))
        sh = base_shape + (max(3, len(streams)),)
        data = np.zeros(sh, dtype=streams[0].dtype)
        for k, a in enumerate(reshape_iter):
            for j, f in enumerate(a):
                data[k,...,j] = f
    return  data, outmeta


def get_zstep(record):
    ch1 = get_field(record[0]['Channel'])
    levels = record[record['Channel']==ch1]['Zlevel']
    return np.mean(np.diff(map(get_field, levels)))


def load_zstack_v7(file_name, record, ch=None):
    record = check_record_type(record, file_name)
    outmeta = {'ch':ch}
    
    dz = get_zstep(record)
    dx = np.mean(map(get_field, record['WidthStep']))
    outmeta['axes'] = lib.alist_to_scale([(dz, 'um'), (dx, 'um')])
    nframes = len(record)
    frame_shape = tuple(get_field(record[0]['DIMS']))
    channels = [get_field(x).lower() for x in record['Channel']]
    nchannels = len(np.unique(channels))
    base_shape = (nframes//nchannels, ) + frame_shape

    var_names = map(get_field, record['ImageName'])
    if ch is None or ch == 'all':
        ## TODO: use smth like np.concatenate(map(lambda m: m.reshape(m.shape+(1,)), (x1,x2,x3)), 3)
        sh = base_shape + (max(3, nchannels),)
        recs = io.loadmat(file_name, variable_names=var_names,appendmat=False)
        stream = lib.clip_and_rescale(np.array([recs[n] for n in var_names]))
        data = np.zeros(sh)
        for k in xrange(nframes//nchannels):
            for j in xrange(nchannels):
                data[k,...,j] = stream[k+j]
    else:
        if type(ch) is int :
            var_names = var_names[ch::nchannels]
        elif type(ch) is str :
            var_names = [n for n,c in zip(var_names,channels) if ch.lower() in c]
        recs = io.loadmat(file_name, variable_names=var_names,appendmat=False)
        data = np.array([recs[n] for n in var_names])
    return data, outmeta

class MES_Record:
    pass

class MES_Timelapse(MES_Record):
    def _reshape_frames(self, stream):
        nlines,nframes = map(int, (self.nlines, self.nframes))
        return (stream[k*nlines:(k+1)*nlines,:] for k in xrange(1,nframes))
    def _load_streams(self,ch=None):
        var_names = self.img_names
        if ch is None:
            ch = self.ch
        if not (ch is None or ch=='all'):
            if isinstance(ch, int):
                var_names = [var_names[ch]]
            elif isinstance(ch, str):
                var_names = [n for n,c
                             in zip(var_names, self.channels)
                             if ch.lower() in c.lower()]
        streams = [self.h5file[n] for n in var_names]
        print var_names
        if len(streams)==0:
            raise IndexError("MES_Timelapse: can't load record%s"%self.recordName)
        return streams
    def load_data(self,ch=None):
        self.ch = ch
        outmeta = dict(ch=self.ch, timestamp=self.timestamps[0])
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

class MES_Timelapse_h5(MES_Timelapse):
    def __init__(self, h5file, recordName, ch=None):
        tkeyf = lambda rec: \
                read_txtentry_h5(h5file, '/'.join((recordName,rec)))
        nkeyf = lambda rec: \
                read_numentry_h5(h5file, '/'.join((recordName,rec)))
        self.h5file = h5file
        self.record = recordName
        self.ch = ch
        self.contexts = np.array(tkeyf('Context'))
        self.channels = tkeyf('Channel')
        self.img_names = np.array(tkeyf('ImageName'))[self.contexts=='Measure']
        self.timestamps = [s for s in tkeyf('MeasurementDate')
                           if s!='\x00\x00']
        self.dims = nkeyf('DIMS')
        #self.line_length = nkeyf('Linelength')
        self.line_length = self.dims[0,0]
        ffi = get_ffi_h5(h5file, recordName)
        self.__dict__.update(ffi)
        self.frame_2d_shape = (self.line_length, self.nlines)
        self.dx = nkeyf('WidthStep')[0]
        self.dt = (self.tstop-self.tstart)/self.nframes


        
        
        

# --- Routines for HDF5 Matlab files (to be refactored later) ----------

import h5py

def field_to_string_h5(h5file, objref):
    return ''.join(unichr(c) for c in h5file[objref])

def read_txtentry_h5(h5file, refstr):
    refs = h5file[refstr]
    return [field_to_string_h5(h5file, e) for e in refs[0,:]]

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
    #timestamps 
    return context_unique
