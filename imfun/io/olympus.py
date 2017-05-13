import re
import numpy as np
from imfun.external.physics import Q

from imfun.core import units
from imfun.core.units import QS

def parse_property(line):
    l = line.strip().replace('"','')
    name, value = l.split('\t')
    values = value.split(',')
    return name, list(map(str.strip, values))

def from_brackets(s):
    result =  re.findall("\[([^]]+)\]",s)
    if result:
        return result[0]
    else:
        return None
    

def parse_header(line):
    return from_brackets(line)

def is_header(line):
    return line.replace('"','')[0] == '['


def parse_dimension_entry(name, values):
    if not  name[0] in 'TZXY':
        return name, values
    np_s, int_s, sampling_s = values
    npoints = int(np_s)
    x = int_s.split()
    units = from_brackets(int_s)
    span = QS(float(x[2])-float(x[0]),units)
    #span = Q(float(x[2])-float(x[0]),units)
    units =  from_brackets(sampling_s)
    if units:
        units = units.split('/')[0]
        s = sampling_s.split()[0]
        #sampling  = pair_to_scale((s,units))
        sampling = QS(float(s), units)
    else:
        sampling = sampling_s
    return name, (npoints, span, sampling)
        
def parse_meta_general(path):
    with open(path) as p:
        lines = p.readlines()
        out = {}
        current_header = None
        for line in lines:
            if is_header(line):
                current_header = parse_header(line)
                out[current_header] = {}
            elif current_header is not None:
                key, values = parse_property(line)
                if current_header == 'Dimensions':
                    key,values = parse_dimension_entry(key,values)
                out[current_header][key] = values
    out['axes'] = dimensions_to_axes(out['Dimensions'])
    return out
    
    
def dimensions_to_axes(md):
    keys = list(md.keys())
    order = ('TZ','Y','X')
    keys = [[k for k in keys if k[0] in o][0] for o in order]
    values = [md[k] for k in keys]
    #out = [isinstance(v[-1],Q) and v[-1] or v[1]/v[0] for v in values]
    out = [QS(v[1].value/v[0],v[1].unit) if isinstance(v[-1],str) else v[-1] for v in values]
    return np.array(out)
