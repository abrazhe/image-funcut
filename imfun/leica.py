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

class LeicaProps_old:
    def __init__(self,xmlfilename):
        import libxml2
        doc = libxml2.parseFile(get_xmljob(xmlfilename))
        for node in doc.xpathEval("//ScannerSettingRecord"):
            Id = get_prop(node, 'Identifier')
            if Id == 'nDelayTime_ms':
                self.dt = float(get_prop(node, 'Variant'))/1e3
        for node in doc.xpathEval("//DimensionDescription"):
            Id = get_prop(node, 'DimID')
            Units = get_prop(node, 'Unit')
            val = 1e6*float(get_prop(node, 'Length')) # um
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
        
        
# ==============

#from xml.sax import saxutils
import xml.sax
from xml.sax import make_parser
from xml.sax.handler import feature_namespaces

class FindVal(xml.sax.ContentHandler):
    names = ['ScannerSettingRecord', 'DimensionDescription', 'TimeStamp']
    def __init__(self, header, value):
        self.search_title, self.search_value = header, value
        self.res = {}
        self.timestamps = []
    def startElement(self, name, attrs):
        # If it's not a needed element, ignore it
        if name not in self.names : return
        title = attrs.get('Identifier', None)
        value = attrs.get('Variant', None)
        if title == 'nDelayTime_ms':
            self.res['dt'] = float(value)/1e3
        elif name == 'DimensionDescription':
            dimid = attrs.get('DimID')
            num_elem = float(attrs.get('NumberOfElements'))
            length = float(attrs.get('Length'))*1e6
            unit = attrs.get('Unit',None)
            if unit == 'm':
                if dimid == '1':
                    self.res['xdim'] = length
                    self.res['dx'] = length/num_elem
                elif dimid == '2':
                    self.res['ydim'] = length
                    self.res['dy'] = length/num_elem
        elif name == 'TimeStamp':
            high = long(attrs.get('HighInteger',None))
            low = long(attrs.get('LowInteger',None))
            self.timestamps.append(ticks_to_ms(low,high)/1000)
    def endDocument(self):
        self.res['start_time'] = self.timestamps[0]
        self.res['stop_time'] = self.timestamps[-1]
        #print self.res
            

def leica_parser():
    parser = make_parser()
    parser.setFeature(feature_namespaces, 0)
    dh = FindVal('nDelayTime_ms', '62')
    parser.setContentHandler(dh)
    return parser


class LeicaProps():
    def __init__(self, xmlfilename):
        p = leica_parser()
        fname = get_xmljob(xmlfilename)
        p.parse(fname)
        res = p.getContentHandler().res
        self.__dict__.update(res)
        
        
            
                
        
                              
