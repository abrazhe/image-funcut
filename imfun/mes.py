# Routines and classes to read MES files (as produced by Femtonics microscopes)

from __future__ import division

import itertools as itt
import numpy as np
from scipy import io
import lib
import fnutils as fu


def guess_format(file_name):
    """given file name, return either 'mat' or 'h5' depending on whether the
    file format is Matlab v5 or Matlab > v7, which is HDF5
    """
    result = None
    with open(file_name, 'r') as fid:
        magic = fid.read(19)
        if 'MATLAB' in magic and '5.' in magic:
            result = 'mat'
        elif 'MATLAB' in magic and '7.' in magic:
            result = 'h5'
        else:
            print "Unknown MES file format"
    return result

def load_file_info(file_name):
    """Load basic information about a file returns a list of MES_Record
    instances (subclassed as MAT_Record or H5_Record depending on file format
    """
    variant = guess_format(file_name)
    if variant == 'mat':
        vars = [x[0] for x in io.whosmat(file_name)
                     if 'Df' in x[0]]
        handler = MAT_Record
    elif variant == 'h5':
        with h5py.File(file_name) as f:
            vars = [k for k in f.keys() if 'Df' in k]
        handler = H5_Record
    else:
        print "Can't load description"
        vars = None
    records = [handler(file_name, v) for v in vars]
    return records

def load_record(file_name, recordName, ch=None):
    """Given file name and a record name, load the record data dispatching on
    the correct file format and record type, i.e. Zstack or Timelapse

    returns one of {ZStack_mat, ZStack_h5, Timelapse_mat or Timelapse_h5}
    """
    valid_records = load_file_info(file_name)
    #valid_names = [r.record for r in valid_records]
    r = filter(lambda r: recordName == r.record, valid_records)
    if len(r) == 0:
        print "Can't find record {} in file {}".format(recordName, file_name)
        print "valid records:", valid_records
        print "WARNING: falling back to first available record"
        r = [valid_records[0]]
        recordName = r[0].record
        
    handlers = {'matZ':ZStack_mat,
                'matT':Timelapse_mat,
                'h5Z':ZStack_h5,
                'h5T':Timelapse_h5}
    r = r[0]
    key = r.variant+r.get_kind()

    print key
    if not 'U' in key:
        obj = handlers[key](file_name, recordName, ch)
    else:
        print "Unknown type of file or record"
        obj = None
    return obj
    

    

## TODO use [()] together with squeeze_me argument for scipy.io.loadmat

def only_measures_mat(entry):
    return entry[entry['Context']=='Measure']

def first_measure_mat(entry):
    measures = only_measures_mat(entry)
    if len(measures):
        return measures[0]

class MES_Record:
    "Base class for a MES Record. Format-specific details are in daughter classes"
    def __repr__(self):
        tstamp = self.timestamps[0].replace(' ','.')
        tstamp = ':'.join(tstamp.split(':')[:-1])
        kind = self.get_kind()
        return '<%s '%self.record + ' '.join((kind,tstamp))+'>'
    def is_supported(self):
        return self.is_zstack() or self.is_timelapse()
    def get_kind(self):
        kind = 'U'
        if self.is_zstack():
            kind = 'Z'
        elif self.is_timelapse():
            kind = 'T'
        #self.kind = kind
        return kind

class MAT_Record(MES_Record):
    """Base functions to read entries from MES-related MATv7 files"""
    def __init__(self, file_name, recordName, ch=None):
        self.variant = 'mat'
        self.ch = ch
        self.record = recordName
        self.entry = io.loadmat(file_name, variable_names=[recordName])[recordName]
        self.contexts = self.entry["Context"]
        self.file_name = file_name
        self.dx = self.get_field(self.entry['WidthStep'])
        self.timestamps = [x[0][0] for x in self.entry['MeasurementDate']
                           if np.prod(x[0].shape)>0 ]
        self.img_names = map(self.get_field, self.entry['ImageName'])
        self.channels = [self.get_field(x).lower() for x in self.entry['Channel']]

    def _get_stream(self, name):
        rec = io.loadmat(self.file_name,variable_names=[name],appendmat=False)
        return rec[name]    
    def _get_recs(self, names):
        return io.loadmat(self.file_name, variable_names=names,appendmat=False)
    def is_zstack(self):
        "Check if an entry is a z-stack"
        entry = self.entry
        return 'Zstack' in entry['Context'] and 'Zlevel' in entry.dtype.names
    def is_timelapse(self):
        """Check if an entry is an XYT measurement. It must have a 'FoldedFrameInfo' propery"""
        return 'FoldedFrameInfo' in self.entry.dtype.names
    def get_field(self, field):
        return field[0][0]
        
        

class H5_Record(MES_Record):
    "Base functions to read entries from MES-related HDF5 MAT files"
    def __init__(self, file_name, recordName,ch=None):
        self.variant = 'h5'
        self.ch = ch
        self.h5file = h5py.File(file_name)
        tkeyf = self._get_str_field
        nkeyf = self._get_num_field
        self.record = recordName
        self.img_names = np.array(tkeyf('ImageName'))
        self.contexts = np.array(tkeyf('Context'))
        self.channels = np.array([s.lower() for s in tkeyf('Channel')])
        self.nchannels = len(np.unique(self.channels))
        self.timestamps = [s for s in tkeyf('MeasurementDate')
                           if s!='\x00\x00']
        self.dims = nkeyf('DIMS')
        self.dx = nkeyf('WidthStep')[0]
    def is_zstack(self):
        return 'Zstack' in self.contexts and\
               'Zlevel' in self.h5file[self.record]
    def is_timelapse(self):
        return 'FoldedFrameInfo' in self.h5file[self.record]
    def _get_stream(self, name):
        return self.h5file[name]
    def _get_recs(self, names):
        return {n:self.h5file[n] for n in names}
    def _get_num_field(self, rec):
        return read_numentry_h5(self.h5file, '/'.join((self.record,rec)))
    def _get_str_field(self, rec):
        return read_txtentry_h5(self.h5file, '/'.join((self.record,rec)))


class ZStack:
    "Common class to deal with Zstack data"
    kind = 'Zstack'
    def _read_frame(self, num, ch=None):
        if ch is None: ch = self.ch
        k =  num*self.nchannels
        pipeline = fu.flcompose(self._get_stream, lambda a: 1.0*np.array(a),
                                lib.rescale)
        names = self.img_names[k:k+self.nchannels][self._ch2ind(self.ch)]

        cstack = np.array(map(pipeline, names))
        return np.squeeze(np.dstack(cstack))
    def __getitem__(self, val):
        indices = np.arange(self.base_shape[0])[val]
        print indices, np.ndim(indices)
        if np.ndim(indices) < 1:
            return self._read_frame(indices)
        return np.array(map(self._read_frame, indices))
        
    def _ch2ind(self, ch):
        if ch is None or ch == 'all':
            out = slice(0,self.nchannels)
        elif isinstance(ch, int):
            out = [ch]
        elif isinstance(ch,basestring):
            out = np.where([ch in s for s in 'rgb'])[0][()]
        return out
                          
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
            if isinstance(ch,int) :
                var_names = var_names[ch::nchannels]
            elif isinstance(ch,basestring) :
                var_names = [n for n,c in zip(var_names,self.channels) if ch.lower() in c]
            recs = self._get_recs(var_names)
            data = np.array([recs[n] for n in var_names])
        return data, outmeta

class ZStack_mat(ZStack, MAT_Record):
    def __init__(self, file_name, recordName, ch=None):
        MAT_Record.__init__(self, file_name, recordName, ch)
        self.dz = self._get_zstep()
        self.nframes = len(self.entry)
        self.frame_shape = tuple(self.get_field(self.entry[0]['DIMS']))
        self.nchannels = len(np.unique(self.channels))
        self.base_shape = (self.nframes//self.nchannels, ) + self.frame_shape
    def _get_zstep(self):
        ch1 = self.get_field(self.entry[0]['Channel'])
        levels = self.entry[self.entry['Channel']==ch1]['Zlevel']
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

class Timelapse:
    "Common class to deal with timelapse images"
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
            #streams = [lib.clip_and_rescale(s) for s in streams]
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
            elif isinstance(ch, basestring):
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
        self.measures = only_measures_mat(self.entry)
        self.dt = self.get_sampling_interval()
        nframes, (line_len, nlines) = self.get_xyt_shape()
        self.img_names = [x[0] for x in self.measures['ImageName']]
        self.channels = [x[0] for x in self.measures['Channel']]
    def _reshape_frames(self, stream):
        nlines,nframes = map(int, (self.nlines, self.nframes))
        return (stream[:,k*nlines:(k+1)*nlines].T for k in xrange(1,nframes))
    def get_xyt_shape(self,):
        'return (numFrames, (side1,side2))'
        m = first_measure_mat(self.entry)
        self.nlines = int(m['FoldedFrameInfo']['numFrameLines'][0])
        self.nframes = int(m['FoldedFrameInfo']['numFrames'][0])
        self.line_length = m['DIMS'][0][0]
        return self.nframes, (self.line_length, self.nlines)
    def get_sampling_interval(self):
        ffi = self.get_ffi()
        #nframes = long(ffi['numFrames'])
        #tstart = float(ffi['firstFrameStartTime'])
        #tstop = float(ffi['frameTimeLength'])
        #return (tstop-tstart)/nframes
        return float(ffi['frameTimeLength'])/1000.
    def get_ffi(self): 
        return first_measure_mat(self.entry)['FoldedFrameInfo'][0]

class Timelapse_h5(Timelapse, H5_Record):
    def __init__(self, file_name, recordName, ch=None):
        H5_Record.__init__(self, file_name, recordName, ch)
        self.img_names = self.img_names[[self.contexts=='Measure']]
        self.line_length = self.dims[0,0]
        ffi = get_ffi_h5(self.h5file, recordName)
        self.__dict__.update(ffi)
        self.frame_2d_shape = (self.line_length, self.nlines)
        self.dt /= 1000. # convert to seconds
        #self.dt = (self.tstop-self.tstart)/self.nframes
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
             ('dt', 'frameTimeLength'),
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
