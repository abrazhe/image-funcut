import numpy as np


from . import mes
from . import leica
from . import ioraw as ioraw
from . import olympus


def write_table_csv(table, fname):
    import csv
    with open(fname, 'w') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        for row in table:
            writer.writerow(row)

def write_dict_hdf(dict, fname):
    print 'write_dict_hdf not supported yet'
    return


def _prepare_dict_for_tabular(dict):
    keys = sorted(dict.keys())
    vals = []
    ext_keys = []
    for k in keys:
        v = dict[k]
        if np.ndim(v)==1:
            vals.append(v)
            ext_keys.append(k)
        elif np.ndim(v) ==2:
            # stack columns
            for i, col in enumerate(v.T):
                vals.append(col)
                ext_keys.append('%s:%d'%(k,i+1))
        else :
            print "can't write values with ndim>2"
    return ext_keys, vals


def write_dict_tab(dict, fname, index=None, topheader=None):
    "write a dictionary (each value is 1D timeseries) to tab-delimited file"
    out_string = ""
    if topheader is not None:
        out_string += topheader
    if index is None:
        index_name = ''
        index_vals = np.arange(Lmax)
    else:
        index_name, index_vals = index

    keys, vals = _prepare_dict_for_tabular(dict)
    Lmax = np.max([len(v) for v in vals] + [len(index_vals)])

    out_string += '\t'.join([index_name]+keys)+'\n'
    for k in range(Lmax):
        out_string += '\t'.join(['{}'.format(index_vals[k])]+
                                ['%e'%v[k] for v in vals])+'\n'

    with open(fname, 'w') as f:
        f.write(out_string)
    return


def write_dict_csv(dict, fname, index=None, **kwargs):
    import csv
    with open(fname, 'w') as f:
        #writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer = csv.writer(f, **kwargs)
        keys, vals = _prepare_dict_for_tabular(dict)
        if index is None:
            index_name = ''
            index_vals = np.arange(Lmax)
        else:
            index_name, index_vals = index
            
        Lmax = np.max([len(v) for v in vals] + [len(index_vals)])
        
        writer.writerow([index_name]+ keys)
        for i in xrange(Lmax):
            row = [index_vals[i]] + [v[i] if len(v) >i and v[i] is not None and not np.isnan(v[i]) else 'NA' for v in vals]
            writer.writerow(row)
