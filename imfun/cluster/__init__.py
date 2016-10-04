"""
A small collection of clustering algorithms. Some algorithms are in a naive implementation.
More mature implementations can be found in `scikit-learn`.
"""

import numpy as np
from numba import jit

from .conv_squares_ import csq_find_rois, csq_plot_rois
from .dbscan_ import dbscan
from . import metrics
#from . import utils

from .kmeans_ import kmeans

from . import som_
from .som_ import som


###---------------------------------------------------------
###             Simple helper functions                  ###
###---------------------------------------------------------
from .utils import sort_clusters_by_size

def quality_threshold(points, max_diam, dist_fn = metrics.euclidean):
    # unfinished, unoptimized, slow, but works
    def _qt_step():
        print "\n"
        clusters = []
        for j,point in enumerate(points): # this can be done in parallel
            clusters.append(_grow_cluster(point))
            sys.stderr.write('\r point %04d'%(j+1))
        x = [c.mass() for c in clusters]
        print "step done"
        return clusters[np.argmax(x)]
    def _grow_cluster(point):
        newc = Cluster([point], dist_fn=dist_fn)
        pointsx = [p for p in points if p != point]
        d = 0
        while d <  max_diam :
            #pointsx = [p for p in points if p not in newc.points]
            x = [newc.farthest_linkage(p) for p in pointsx]
            k = np.argmin(x)
            d = x[k]
            newc.addpoint(pointsx[k])
            pointsx.pop(k)
            if len(pointsx) < 2: break
        return newc
    out_clusters = []
    while len(points)>2:
        sys.stderr.write("\r %04d points remain"%len(points))
        cluster = _qt_step()
        out_clusters.append(cluster)
        points = [p for p in points if p not in cluster.points]
    return out_clusters
