# Very simple functions to extract useful data from Leica Microscope
# XML files

def get_prop(node, prop):
    return node.xpathEval('@%s'%prop)[0].content

import libxml2

class LeicaProps:
    def __init__(self,xmlfilename):
        doc = libxml2.parseFile(xmlfilename)
        a = doc.xpathEval("//ScannerSettingRecord")
        for node in a:
            Id = get_prop(node, 'Identifier')
            if Id == 'nDelayTime_ms':
                self.dt = float(get_prop(node, 'Variant'))/1e3
            elif Id == 'dblSizeX':
                self.xscale = float(get_prop(node,'Variant'))*1e6
            elif Id == 'dblSizeY':
                self.yscale = float(get_prop(node,'Variant'))*1e6
        
                              
