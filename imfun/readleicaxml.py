# Very simple functions to extract useful data from Leica Microscope
# XML files

import glob

def get_prop(node, prop):
    return node.xpathEval('@%s'%prop)[0].content



def get_xmljob(name, patt = "*[0-9].xml"):
    if name[-4:] == '.xml':
        return name
    if name[-1] != '/':
        name += '/'
    return glob.glob(name+patt)[0]

def ticks_to_ms(lo,hi):
    """
    Converts from two-word tick representation to milliseconds.
    idea tacken from DateTools.java from loci:
    http://skyking.microscopy.wisc.edu/svn/java/trunk/components/common/src/loci/common/DateTools.java
    """
    ticks = (long(hi)<<32)|long(lo)
    return ticks/10000.0 # 100 ns = 0.0001 ms

def low_high_pair(a):
    return map(long, (get_prop(a, "LowInteger"),
                      get_prop(a, "HighInteger")))

class LeicaProps:
    import libxml2
    def __init__(self,xmlfilename):
        doc = libxml2.parseFile(get_xmljob(xmlfilename))
        for node in doc.xpathEval("//ScannerSettingRecord"):
            Id = get_prop(node, 'Identifier')
            if Id == 'nDelayTime_ms':
                self.dt = float(get_prop(node, 'Variant'))/1e3
        for node in doc.xpathEval("//DimensionDescription"):
            Id = get_prop(node, 'DimID')
            Units = get_prop(node, 'Unit')
            val = float(get_prop(node, 'Length'))
            num_elem = float(get_prop(node, 'NumberOfElements'))
            if Units is 'm':
                if Id is '1':
                    self.xdim = val
                    self.dx = val/num_elem
                elif Id is '2':
                    self.ydim = val
                    self.dy = val/num_elem
        tstamps = doc.xpathEval("//TimeStamp")
        self.start_time = ticks_to_ms(*low_high_pair(tstamps[0]))
        self.stop_time = ticks_to_ms(*low_high_pair(tstamps[-1]))
        
        
            
            
            
            
                
        
                              
