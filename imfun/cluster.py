### Simple clustering (k-means and such).
###
### Reimplementing (naive implementation) it here mainly for educational purposes.
### This has to be re-done using structures for inter-point distances
import sys
import numpy as np
import random as pyrand
import itertools as itt
from scipy import stats


### ---  Distance measures ---
def minkowsky(p1,p2,k):        
    if (type(p1) == np.ndarray) and (type(p2) == np.ndarray):
        return np.power(np.sum(abs((p1-p2))**k), 1./k)
    else:
        x = map(lambda x,y: abs((x-y)**k), p1, p2)
        return np.power(np.sum(x), 1./k)

def euclidean(p1,p2):
    return minkowsky(p1,p2,2)

def cityblock(p1,p2):
    return minkowsky(p1,p2,1)

def pearson(v1,v2):
    return 1 - stats.pearsonr(v1,v2)[0]

def apearson(v1,v2):
    return 1 - abs(stats.pearsonr(v1,v2)[0])

def spearman(v1,v2):
    return 1 - stats.spearmanr(v1,v2)[0]

def xcorrdist(v1,v2):
    return 1/np.correlate(v1,v2)


## TODO: distance_matrix

##---------------------------

## TODO: remove class point, use just list or tuple
class Point:
    def __init__(self, coords,ref=None):
        self.n = len(coords)
        self.pos = coords
    def __repr__(self):
        return str( self.pos)
    
class Cluster:
    def __init__(self, points,
                 center_fn = np.mean,
                 dist_fn = euclidean):
        self.center_fn = center_fn
        self.dist_fn = dist_fn
        self.set_points(points)
    def __repr__(self):
        return str(self.points)
    def set_points(self, points):
        if len(points) < 1:
            raise Exception("Cluster problem: \
            each cluster should have at least one point")
        self.points = points
    def distortion(self):
        c = self.centroid()
        return np.sum(map(lambda x: self.dist_fn(c,x),
                          self.points))
    def distortion2(self):
        c = self.centroid()
        return np.mean(map(lambda x: self.dist_fn(c,x),
                          self.points))
    def diam(self):
        if len(self.points) > 1:
            return np.max([self.dist_fn(*pair)
                           for pair in itt.combinations(self.points,2)])
        else:
            return 0
    def farthest_linkage(self, point):
        return np.max([self.dist_fn(p,point) for p in self.points])
    def addpoint(self, point):
        self.points.append(point)
    def update(self, points):
        old_center = self.centroid()
        self.set_points(points)
        new_center = self.centroid()
        return self.dist_fn(old_center, new_center)
    def centroid(self,):
        pcoords = [p for p in self.points]
        return self.center_fn(pcoords,0)
    def mass(self):
        return len(self.points)
    pass

def arreq(arr1,arr2):
    return np.all(np.equal(arr1,arr2))

def select_points(points, affiliations, idx):
    return [p for a,p in zip(affiliations, points) if a == idx]

def kmeans1(points, k, tol=1e-3,
            center_fn = np.mean,
            distance = euclidean,
            max_iter = 1e7,
            output = 'indices',
            verbose = True):
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
    elif output== 'indices':
        return affiliations
    elif output == 'full':
        return clusters, affiliations
        




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
def dbscan(points, eps, min_pts, dist_fn = euclidean,
           verbose=False):
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

#expandCluster(P, N, C, eps, MinPts)
#   add P to cluster C
#   for each point P' in N 
#      if P' is not visited
#         mark P' as visited
#         N' = getNeighbors(P', eps)
#         if sizeof(N') >= MinPts
#            N = N joined with N'
#      if P' is not yet member of any cluster
#         add P' to cluster C

### here i tried with pre-calculation of distances between points
### for dbscan. Didn't help the speed though
def distance_dict_redun(points, dist_fn=euclidean):
    "redundant mapping of pairwise distances"
    dd = {}
    for p in points: dd[p] = {}
    for p1,p2 in itt.permutations(points,2):
        dd[p1][dist_fn(p1,p2)] = p2
    return dd

def kdist(points, k, dist_fn = euclidean):
    dd = distance_dict_redun(points, dist_fn)
    dists = []
    for point in points:
        x = sorted(dd[point].keys())
        dists.append(x[k])
    return sorted(dists, reverse=True)

def distance_dict(points, dist_fn = euclidean):
    "non-redundant mapping of distances"
    dd = {}
    for pair in itt.combinations(points,2):
        dd[pair] = dist_fn(*pair)
    return dd

def alldistances_test1(points, dist_fn = euclidean):
    for pair in itt.combinations(points,2):
        dist_fn(*pair)
        
def neighbours_dict(points, eps,dist_fn = euclidean):
    nd = {}
    #dd = distance_dict(points,dist_fn)
    for p in points: nd[p] = []
    for p1,p2 in itt.permutations(points,2):
        d =  dist_fn(p1,p2)
        if d <  eps:
            nd[p1].append(p2)
    return nd
###-----------------------------------


###---- converging squares ----------

## a square is a list of slices

def child_slices(sl, step=1):
    "split a slice into two overlapping smaller slices"
    return [slice(sl.start+step,sl.stop),
            slice(sl.start,sl.stop-step)]
    
def child_squares(square, step=1):
    "split a nD square into a list of overlapping smaller squares"
    lcsl = map(lambda s:child_slices(s,step), square)
    return list(itt.product(*lcsl))

def square_size(sq):
    ## as it is assumed a square, only the
    ##  size in first dimension is needed
    return sq[0].stop-sq[0].start

def converge_square(m, square, step=1,
                   efunc=np.sum, min_size=1):
    if square_size(square) > min_size:
        chsq = child_squares(square)
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
    find regions of interest with converging squares
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
    import pylab as pl
    pl.figure()
    pl.imshow(m, aspect='equal', cmap='gray')
    positions = [[[s.start] for s in r[::-1]] for r in rois]
    points = map(csqroi2point, rois)
    for p in points:
        pl.plot(*p,ls='none',color='r',marker='s')

def csqroi2point(roi):
    return [s.start for s in roi[::-1]] # axes are reverse to indices
    

def make_grid(shape,stride):
    origins =  itt.product(*[range(0,dim,stride) for dim in shape])
    squares = ([slice(a,a+stride) for a in o] for o in origins)
    return squares

###-----------------------------------

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
    """
    all locations for a shape; substitutes nested cycles
    """
    return itt.product(*map(xrange, shape))

    
def mask2points(mask):
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

def surfconvert(frame):
    out = []
    nr,nc = map(float, frame.shape)
    f = lib.rescale(frame)
    for r in range(int(nr)):
        for c in range(int(nc)):
            out.append([c/nc,r/nr, f[r,c]])
    return np.array(out)



###### Now, use mdp (no good so far)
try:
    import mdp
    def do_gng(arr,**kwargs):
        gng = mdp.nodes.GrowingNeuralGasNode(**kwargs)
        gng.train(arr)
        gng.stop_training()
        return gng


    def plot_mdp_nodes(arr, gng):
        #figure(figsize=(6,6));
        #scatter(*arr.T, color='k', s = 0.3)
        colors = ['r', 'b','g','c','m','y']
        objs = gng.graph.connected_components()
        for j,o in enumerate(objs):
            coords = np.array([node.data.pos for node in o])
            scatter(coords[:,1],coords[:,2],color=colors[j%len(colors)])
except:
    pass
        
