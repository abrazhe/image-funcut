import numpy as np
from .Cluster_ import Cluster
from . import metrics

def kmeans(points, k, tol=1e-3,
            center_fn = np.mean,
            distance = metrics.euclidean,
            max_iter = 1e7,
            output = 'labels',
            verbose = True):
    '''Simple K-means implementation

    Parameters:
      - `points`: (collection) -- a collection of points, each point is a
                  list/tuple/array of coordinates
      - `k`: (number) -- number of clusters
      - `tol`: (number) -- minimal shift of cluster centroid, before stop of algorithm
      - `center_fn`: (function) -- a function to find centroid; by default, ``numpy.mean``
      - `distance`: (function) -- a distance measure; by default, ``euclidean``]
      - `max_iter`: (number) -- maximal number of iterations
      - `output`: (string) -- can be "labels", "clusters" and "full", modifies
                  return value
      - `verbose`: (bool) -- if ``True``, be verbose

    Returns: modified by `output` argument
      - if `output` is 'labels', return a list of integer labels, specifying
        point affiliation to a cluster
      - if `output` is 'clusters' return clusters
      - if `ouput` is 'full' return a tuple of clusters and affiliations
    '''
    #affiliations = np.random.randint(k, size=len(points))
    affiliations = np.ones(len(points))*-1
    random_pidx = np.random.randint(len(points),size= k)
    print(random_pidx)
    clusters = [Cluster([points[i]],
                        center_fn = center_fn,
                        dist_fn = distance)
                for i in random_pidx]
    niter = 0
    updater = [[] for i in range(k)]
    shifts = [0]*k
    while niter < max_iter:
        reassigned = 0
        affiliations_prev = affiliations.copy()
        for i in range(k): updater[i] = []
        centers = [c.centroid() for c in clusters]
        for j,p in enumerate(points):
            ind = np.argmin([distance(p,c) for c in centers])
            updater[ind].append(p)
            affiliations[j] = ind
        for j,c in enumerate(clusters):
            u = updater[j]
            c.set_points(u)
            shifts[j] = distance(centers[j], c.centroid())
        reassigned = np.sum(np.not_equal(affiliations, affiliations_prev))
        print("Iteration %d, reassigned %d points, max shift %f"%(niter,
                                                                  reassigned,
                                                                  np.max(shifts)))
        if reassigned == 0 or (np.max(shifts) < tol):
            break
        niter +=1
    if niter >= max_iter:
        print("Warning: maximum number of iterations reached")
    clusters.sort(key = lambda x: x.mass(), reverse=True)
    if output=='clusters':
        return clusters
    elif output== 'labels':
        return affiliations
    elif output == 'full':
        return clusters, affiliations
    else:
        print("""'output' must be one of 'clusters', 'indices', 'full', returning clusters""")
        return clusters
        
