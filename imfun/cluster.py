### Simple clustering (k-means and such).
###
### Reimplementing (naive implementation) it here mainly for educational purposes.
### This has to be re-done using structures for inter-point distances
import sys
import numpy as np
#import random as pyrand
import itertools as itt
from scipy import stats


### ---  Distance measures ---
def minkowski(p1,p2,k):
    '''Minkowski distance between points p1 and p2
    if k equals 1, it is a Manhattan (cityblock) distance
    if k equals 2, it is a Euclidean distance
    '''
    if  isinstance(p1, np.ndarray) and isinstance(p2, np.ndarray):
        return np.power(np.sum(np.abs((p1-p2)).T**k, 0), 1./k) # do axis sum
                                        # to allow for vectorized input
    else:
        x = map(lambda x,y: abs((x-y)**k), p1, p2)
        return np.power(np.sum(x), 1./k)

def euclidean(p1,p2):
    "Euclidean distanc between 2 points"
    return minkowski(p1,p2,2)

def cityblock(p1,p2):
    "Cityblock (Manhattan) distance between 2 points"
    return minkowski(p1,p2,1)

def pearson(v1,v2):
    "Pearson distance measure"
    return 1 - stats.pearsonr(v1,v2)[0]

def apearson(v1,v2):
    "Absolute Pearson distance measure"
    return 1 - abs(stats.pearsonr(v1,v2)[0])

def spearman(v1,v2):
    "Spearman distance measure"
    return 1 - stats.spearmanr(v1,v2)[0]

def xcorrdist(v1,v2):
    "Correlation distance measure"
    return 1/np.correlate(v1,v2)


## TODO:
### [X] distance_matrix
### [ ] sparse distance matrix
### [ ] rename dist_fn to metrics
### [ ] use scipy.spatial for distance calculations



##---------------------------

## TODO: remove class point, use just list or tuple
class Point:
    def __init__(self, coords,ref=None):
        self.n = len(coords)
        self.pos = coords
    def __repr__(self):
        return str( self.pos)
    
class Cluster:
    '''
    Cluster representatin class
    '''
    def __init__(self, points,
                 center_fn = np.mean,
                 dist_fn = euclidean):
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
        return np.sum(map(lambda x: self.dist_fn(c,x),
                          self.points))
    def distortion_mean(self):
	"mean of distances of all points from the centroid"
        c = self.centroid()
        return np.mean(map(lambda x: self.dist_fn(c,x),
                          self.points))
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

def arreq(arr1,arr2):
    '''elemen-wise equality of two arrays'''
    return np.all(np.equal(arr1,arr2))

def select_points(points, affiliations, idx):
    return [p for a,p in zip(affiliations, points) if a == idx]

def kmeans1(points, k, tol=1e-3,
            center_fn = np.mean,
            distance = euclidean,
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
    print random_pidx
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
        for i in xrange(k): updater[i] = []
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
        print "Iteration %d, reassigned %d points, max shift %f"%(niter,
                                                                  reassigned,
                                                                  np.max(shifts))
        if reassigned == 0 or (np.max(shifts) < tol):
            break
        niter +=1
    if niter >= max_iter:
        print "Warning: maximum number of iterations reached"
    clusters.sort(key = lambda x: x.mass(), reverse=True)
    if output=='clusters':
        return clusters
    elif output== 'labels':
        return affiliations
    elif output == 'full':
        return clusters, affiliations
    else:
	print """'output' must be one of 'clusters', 'indices', 'full', returning clusters"""
	return clusters
        
def quality_threshold(points, max_diam, dist_fn = euclidean):
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

## note: can't work with points as ndarrays because
## 1. they are not hashable (convert to tuples) and
## 2. can't be compared simply by p1 == p2.
def dbscan_old(points, eps, min_pts, dist_fn = euclidean,
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

## here I used dbscan_.py from sklearn as a reference
def dbscan(points, eps, min_pts, distances=None,dist_fn='euclidean',verbose=True):
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
    

        
###-----------------------------------


###---- converging squares ----------

## a square is a list of slices

def _child_slices(sl, step=1):
    "split a slice into two overlapping smaller slices"
    return [slice(sl.start+step,sl.stop),
            slice(sl.start,sl.stop-step)]
    
def _child_squares(square, step=1):
    "split a nD square into a list of overlapping smaller squares"
    lcsl = map(lambda s:_child_slices(s,step), square)
    return list(itt.product(*lcsl))

def square_size(sq):
    ## as it is assumed a square, only the
    ##  size in first dimension is needed
    return sq[0].stop-sq[0].start

def converge_square(m, square, step=1,
                   efunc=np.sum, min_size=1):
    """
    Converge one starting square to a small square

    Parameters:
      - `m`: input ND matrix
      - `square`: a starting structuring element, a list of slices
      - `step` : deflating coefficient
      - `efunc`: a measure function to apply to elements within a square
      - `min_size`: smallest size of the square when the algorithm is stopped

    Returns:
      - final (smallest) square which maximizes the efunc over the elements
    """
    if square_size(square) > min_size:
        chsq = _child_squares(square, step)
        x = [efunc(m[sq]) for sq in chsq]
        return converge_square(m, chsq[np.argmax(x)],
			       step, efunc, min_size)
    else:
	return square # undeflatable

def csq_find_rois(m, threshold = None,
                  stride=5,
                  reduct_step=1, efunc=np.mean,
                  min_size = 1):
    """
    Find regions of interest in an image with converging squares algorithm

    Parameters:
      - `m`: an N-dimensional matrix
      - `threshold`: a threshold whether a local starting square should be
                     taken into account
      - `stride`: size of a starting square
      - `reduct_step`: square is reduced by this step at each iteration
      - `efunc`: a measure function to apply to elements within a square
                 [``np.sum``]
      - `min_size`: smallest size of the square when the algorithm is stopped

    Returns:
      - a list of found ROIs as minimal squares
    
    """
    if threshold is None:
        threshold = np.std(m)
    cs = lambda s: converge_square(m,s,reduct_step,efunc,min_size)
    rois = []
    for square in make_grid(m.shape, stride):
        if efunc(m[square]) > threshold:
            rois.append(cs(square))
    return rois

def plot_csq_rois(m,rois):
    """
    A helper function to plot the ROIs determined by the
    csq_find_rois implementation of the converging squares algorithm

    Parameters:
      - `m` : a 2D matrix
      - `rois`: a list of ROIs
    """
    import pylab as pl
    pl.figure()
    pl.imshow(m, aspect='equal', cmap='gray')
    positions = [[[s.start] for s in r[::-1]] for r in rois]
    points = map(csqroi2point, rois)
    for p in points:
        pl.plot(*p,ls='none',color='r',marker='s')

def csqroi2point(roi):
    """Helper function, converts a ROI to a point to plot"""
    return [s.start for s in roi[::-1]] # axes are reverse to indices
    

def make_grid(shape,stride):
    """Make a generator over sets of slices which go through the provided shape
       by a stride
    """
    origins =  itt.product(*[range(0,dim,stride) for dim in shape])
    squares = ([slice(a,a+stride) for a in o] for o in origins)
    return squares

###---------------------------------------------------------
###             Simple helper functions                  ###
###---------------------------------------------------------


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



## ###### Now, use mdp (no good so far)
## try:
##     import mdp
##     def do_gng(arr,**kwargs):
##         gng = mdp.nodes.GrowingNeuralGasNode(**kwargs)
##         gng.train(arr)
##         gng.stop_training()
##         return gng


##     def plot_mdp_nodes(arr, gng):
##         #figure(figsize=(6,6));
##         #scatter(*arr.T, color='k', s = 0.3)
##         colors = ['r', 'b','g','c','m','y']
##         objs = gng.graph.connected_components()
##         for j,o in enumerate(objs):
##             coords = np.array([node.data.pos for node in o])
##             scatter(coords[:,1],coords[:,2],color=colors[j%len(colors)])
## except:
##     pass
        
