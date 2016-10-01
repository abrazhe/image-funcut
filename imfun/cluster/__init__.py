"""
A small collection of clustering algorithms. Some algorithms are in a naive implementation.
More mature implementations can be found in `scikit-learn`.
"""

import numpy as np
from numba import jit

from .conv_squares_ import csq_find_rois, csq_plot_rois
from .dbscan_ import dbscan
from . import metrics
from .kmeans_ import kmeans
from .som_ import som
from . import som_



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







###---------------------------------------------------------
###             Simple helper functions                  ###
###---------------------------------------------------------

def select_points(points, affiliations, idx):
    return [p for a,p in zip(affiliations, points) if a == idx]


def filter_clusters_size(clusters, min_size=100):
    return filter(lambda x: x.mass() > min_size, clusters)

def plot_clusters(points, clusters):
    import pylab as pl
    pl.figure(figsize=(6,6))
    arr = points2array(points)[:,:2]
    pl.scatter(*arr.T[:2,:], color='k', s=1)
    colors = ['r','b','g','c','m','y']
    for j,c in enumerate(clusters):
        pl.scatter(*cluster2array(c).T[:2,:], color=colors[j%len(colors)],
                alpha=0.5)

def plot3_clusters(points, clusters):
    
    from mpl_toolkits.mplot3d import axes3d

    pl.figure(figsize=(6,6))
    ax = pl.axes(projection='3d')
    arr = points2array(points)[:,:3]
    plot(*arr.T[:3,:], color='k', ls='none',
         marker=',', alpha=0.3)
    colors = ['r','b','g','c','m','y']
    for j,c in enumerate(clusters):
        pl.plot(*cluster2array(c).T[:3,:],
             ls = 'none', marker=',',
             color=colors[j%len(colors)],
             alpha=0.5)


def locations(shape):
    """ all locations for a shape; substitutes nested cycles
    """
    return itt.product(*map(xrange, shape))

def mask2points(mask):
    "mask to a list of points, as row,col"
    points = []
    for loc in locations(mask.shape):
        if mask[loc]:
            points.append(loc) 
    return points

    
def mask2pointsr(mask):
    "mask to a list of points, with X,Y coordinates reversed"
    points = []
    for loc in locations(mask.shape):
        if mask[loc]:
            points.append(loc[::-1]) 
    return points

def array2points(arr):
    return [r for r in surfconvert(arr)]
    
def cluster2array(c):
    "helpful for scatter plots"
    return points2array(c.points)

def points2array(points,dtype=np.float64):
    return np.array(points, dtype=dtype)


def surfconvert(frame, mask):
    from imfun import lib
    out = []
    nr,nc = map(float, frame.shape)
    space_scale = max(nr, nc)
    f = lib.rescale(frame)
    for r in range(int(nr)):
        for c in range(int(nc)):
	    if not mask[r,c]:
		out.append([c/space_scale,r/space_scale, f[r,c]])
    return np.array(out)

