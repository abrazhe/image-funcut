"""
A class to represent a cluster of points
"""

import numpy as np
from . import metrics

## TODO: remove class point, use just list or tuple
class Point:
    def __init__(self, coords,ref=None):
        self.n = len(coords)
        self.pos = coords
    def __repr__(self):
        return str( self.pos)
    
class Cluster:
    '''
    Cluster representation class
    '''
    def __init__(self, points,
                 center_fn = np.mean,
                 dist_fn = metrics.euclidean):
        self.center_fn = center_fn
        self.dist_fn = dist_fn
        self.set_points(points)
    def __repr__(self):
        return str(self.points)
    def set_points(self, points):
        "set points that belong to the cluster"
        if len(points) < 1:
            raise Exception("Cluster problem: \
            each cluster should have at least one point")
        self.points = points
    def distortion(self):
        "sum of distances of all points from the centroid"
        c = self.centroid()
        return np.sum([self.dist_fn(c,x) for x in self.points])
    def distortion_mean(self):
        "mean of distances of all points from the centroid"
        c = self.centroid()
        return np.mean([self.dist_fn(c,x) for x in self.points])
    def diam(self):
        "maximal distance between two points in the cluster"
        if len(self.points) > 1:
            return np.max([self.dist_fn(*pair)
                           for pair in itt.combinations(self.points,2)])
        else:
            return 0
    def farthest_linkage(self, point):
        "maximal distance from a given point to points in the cluster"
        return np.max([self.dist_fn(p,point) for p in self.points])
    def addpoint(self, point):
        "add a point to the cluster"
        self.points.append(point)
    def update(self, points):
        '''replace old points with new ones and return the change in centroid position'''
        old_center = self.centroid()
        self.set_points(points)
        new_center = self.centroid()
        return self.dist_fn(old_center, new_center)
    def centroid(self,):
        "return coordinates of the centroid"
        pcoords = [p for p in self.points]
        return self.center_fn(pcoords,0)
    def mass(self):
        "return number of points in the cluster"
        return len(self.points)
    pass
