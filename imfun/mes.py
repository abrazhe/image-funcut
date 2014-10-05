# Routines to read MES files

from __future__ import division

import numpy as np
from scipy import io


def load_meta(file_name):
    "Load meta information from a MES file"
    max_records = 1000
    var_names = ["Df%04d"%k for k in range(1,max_records+1)]
    meta = io.loadmat(file_name, variable_names=var_names,appendmat=False)
    return meta

def describe_file(name):
    meta = load_meta(name)
    print "# File: %s"%name
    print '-'*(len(name)+10)
    keys = record_keys(meta)
    print """## Data structures:"""
    for key in keys:
        print '###', key
        timestamps = [x[0][0] for x in meta[key]['MeasurementDate']
                      if np.prod(x[0].shape)>0 ]
        context = set([c[0][0] for c in meta[key]["Context"]])
        print 'Contexts:', context
        print 'Timestamps (first,last):', \
              timestamps[0], ', ', len(timestamps)>1 and timestamps[-1] or ''

def record_keys(meta):
    return sorted(filter(lambda x: 'Df' in x, meta.keys()))
    
def is_zstack(entry):
    "Check if an entry is a z-stack"
    return 'Zstack' in entry['Context']

def is_xyt(entry):
    "Check if an entry is an XYT measurement"
    return 'Measure' in entry['Context']

def only_measures(entry):
    return entry[entry['Context']=='Measure']

def first_measure(entry):
    measures = only_measures(entry)
    if len(measures):
        return measures[0]

# ! re-do
def get_date(entry):
    return entry['MeasurementDate'][0]

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
    
    
    
