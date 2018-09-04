## here I used dbscan_.py from sklearn as a reference

import numpy as np

from . import metrics
from .Cluster_ import Cluster

def dbscan(points, eps, min_pts, distances=None,dist_fn='euclidean',verbose=True):
    """ Implementation of DBSCAN density-based clustering algorithm

    Parameters:
      - `points`: input collection of points. 
      - `eps`: (`number`) --  neighborhood radius
      - `min_pts`: (`number`) -- minimal number of neighborhood points
      - `dist_fn`: (`function`) -- distance measure, [``euclidean``]
      - `verbose`: (`bool`) -- if ``True``, be verbose

    Returns:
      a list of clusters (each cluster is a Cluster class instance)
    """
    points = np.asarray(points)
    L = len(points)
    if distances is None:
        D = _pairwise_euclidean_distances(points)
    else:
        D = distances
    #neighborhoods = [np.where(x <= eps)[0] for x in D]
    neighborhoods = [np.where(x <= eps)[0] for x in D]
    labels = -np.ones(L) # everything is noise at start
    core_points = []
    label_marker = 0
    perm = np.random.permutation(L) # we iterate through points in random order
    for k in perm: 
        if labels[k] != -1 or len(neighborhoods[k]) < min_pts:
            # either visited or not a core point
            continue
        core_points.append(k)
        labels[k] = label_marker
        # expand cluster
        candidates = [k]
        while len(candidates) > 0:
            new_candidates = []
            for c in candidates:
                noise = np.where(labels[neighborhoods[c]] == -1)[0]
                noise = neighborhoods[c][noise] # ?
                labels[noise] = label_marker
                for neigh in noise:
                    if len(neighborhoods[neigh]) >= min_pts:
                        new_candidates.append(neigh) # another core point
                        core_points.append(neigh)
            candidates = new_candidates
        label_marker += 1
    clusters = sorted([points[labels==lm] for lm in range(label_marker)],
                      key=lambda x: len(x), reverse=True)
    clusters = [Cluster(c, dist_fn = dist_fn) for c in clusters]
    
    return clusters, core_points, labels


def _pairwise_euclidean_distances(points):
    """pairwise euclidean distances between points.
    Calculated as distance between vectors x and y:
    d = sqrt(dot(x,x) -2*dot(x,y) + dot(Y,Y))
    """
    X = np.asarray(points)
    XX = np.sum(X*X, axis=1)[:,np.newaxis]
    D = -2 * np.dot(X,X.T) + XX + XX.T
    np.maximum(D, 0, D)
    # todo triangular matrix, sparse matrix
    return np.sqrt(D)


## note: can't work with points as ndarrays because
## 1. they are not hashable (convert to tuples) and
## 2. can't be compared simply by p1 == p2.
def dbscan_old(points, eps, min_pts, dist_fn = metrics.euclidean,
           verbose=False):
    """ Implementation of DBSCAN density-based clustering algorithm

    Parameters:
      - `points`: input collection of points. points must be hashable
                  (i.e. a point is a tuple of coordinates)
      - `eps`: (`number`) --  neighborhood radius
      - `min_pts`: (`number`) -- minimal number of neighborhood points
      - `dist_fn`: (`function`) -- distance measure, [``euclidean``]
      - `verbose`: (`bool`) -- if ``True``, be verbose

    Returns:
      a list of clusters (each cluster is a Cluster class instance)
    """
    clusters, marks = [], {}
    t = tuple
    for p in points:  marks[t(p)] = 'U' # U, V, N, C
    #-
    def get_neighbours(p):
        return [pn for pn in points
                if (dist_fn(p, pn) < eps)
                and (p != pn)] # here where it fails with arrays
    #-
    def expand_cluster(p, neighs):
        cluster = [p]
        marks[t(p)] = 'C'
        for pn in neighs:
            if marks[t(pn)] == 'U':
                marks[t(pn)] = 'V'
                neighs2 = get_neighbours(pn)
                if len(neighs2) > min_pts:
                    neighs.extend([n for n in neighs2
                                   if (n not in neighs) and (marks[t(n)] != 'N')])
            if marks[t(pn)] != 'C':
                cluster.append(pn)
                marks[t(pn)] = 'C'
        return cluster
    #-
    Npts = len(points)
    for j,point in enumerate(points):
        if verbose:
            sys.stderr.write("\r point %05d out of %d"%(j+1, Npts))
        if marks[t(point)] == 'U':
            marks[t(point)] = 'V'
            Neighs = get_neighbours(point)
            if len(Neighs) > min_pts:
                c = expand_cluster(point, Neighs)
                points = [p for p in points if p not in c] #remove these points 
                clusters.append(c)
            else:
                marks[t(point)] = 'N'
    clusters.sort(key=lambda x: len(x), reverse = True)
    return [Cluster(c, dist_fn = dist_fn) for c in clusters]
