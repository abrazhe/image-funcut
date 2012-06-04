# Prototype of Kohonen's Self-Organizing Maps
import numpy as np
import itertools as itt
from cluster import euclidean, cityblock, pearson, spearman, xcorrdist


import sys


def neigh_gauss(x1,x2, r):
    return np.exp(-euclidean(x1,x2)**2/r**2)

def voronoi_inds(patterns, maps, distance):
    affiliations = np.zeros(len(patterns))
    for k,p in enumerate(patterns):
        affiliations[k] = np.argmin([distance(p, m)
                                     for m in np.flatiter(maps)])
    return affiliations

distance_fns = {
    'euclidean':euclidean,
    'cityblock':cityblock,
    'pearson':pearson,
    'spearman':spearman,
    'xcorrdist':xcorrdist
    }

def som1(patterns, shape=(10,1), alpha=0.99, r=2.0, neighbour_fn=neigh_gauss,
         fade_coeff = 0.9,
         min_reassign=10,
         max_iter = 1e5,
         distance=euclidean,
	 output = 'last',
         verbose = 0):
    """SOM as described in Bacao, Lobo and Painho, 2005"""

    if (type(distance) is str) and distance_fns.has_key(distance):
        distance = distance_fns[distance]
        
    
    niter = 0
    Npts = len(patterns)            # number of patterns
    L = len(patterns[0])            # dimensionality
    grid = np.zeros((shape[0], shape[1], L))
    locs = list(itt.product(*map(xrange,shape)))
    init_ks = np.random.randint(len(patterns), size=len(locs))
    for k,l in enumerate(locs):
        grid[l] = patterns[k]
    affiliations = np.ones(Npts)*-1
    reassigned = len(affiliations)
    out = []
    while alpha > 1e-6 and niter < max_iter and reassigned > min_reassign:
        affiliations_prev = affiliations.copy()
        for k,p in enumerate(patterns):
            if (not k%100) and verbose:
                sys.stderr.write("%04d %06d, %06d \r"%(niter, reassigned, Npts-k))
            dists = [distance(grid[loc],p) for loc in locs]
            winner_ind = np.argmin(dists)
            affiliations[k] = winner_ind
            winner_loc = locs[winner_ind]
            for loc in locs:
                grid[loc] += alpha*neighbour_fn(winner_loc, loc, r)*(p-grid[loc])
        alpha *= fade_coeff
        r *= fade_coeff
        reassigned = np.sum(affiliations != affiliations_prev)
        niter +=1
	out.append(affiliations.copy())
    if output == 'last':
	out = affiliations
    return out

def som_batch(patterns, shape=(10,1), neighbour_fn = neigh_gauss,
              distance=euclidean):
    print "Not implemented yet"
    pass

def cluster_map_permutation(affs, perms, shape):
    "auxiliary function to map affiliations to 2D image"
    import itertools as itt
    out = np.zeros(shape)
    coordinates = list(itt.product(*map(xrange,shape)))
    for k,a in enumerate(affs):
        out[coordinates[perms[k]]] = a
    return out
