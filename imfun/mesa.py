# Routines to read MESA files

from scipy import io

def load_meta(file_name):
    "Load meta information from a MES file"
    max_records = 1000
    var_names = ["Df%04d"%k for k in range(1,max_records+1)]
    meta = io.loadmat(file_name, variable_names = var_names)
    return meta

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
    return entry[entry['Context']=='Measure'][0]

def get_date(entry):
    return first_measure(entry)['MeasurementDate'][0]

def get_ffi(entry):
    return first_measure(entry)['FoldedFrameInfo'][0]

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
    
    
    
