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

def _sorted_affs(affs):
    return np.array(sorted(range(int(np.max(affs)+1)), key=lambda i:np.sum(affs==i)))


def _iterative_som(patterns, max_rec=50, *args, **kwargs):
    affs_p, gr_p = som1(patterns, *args, output='both',  **kwargs)
    hist = []
    for count in xrange(max_rec):
	affs_n, gr_n = som1(patterns, *args, output='both',
			    init_templates = [g[0] for g in gr_p],
			    **kwargs)
	err = np.sum(abs(gr_n-gr_p))
	
	if np.allclose(_sorted_affs(affs_p), _sorted_affs(affs_n)):
	    return affs_n, gr_n
	hist.append(err)
	print count, np.sum(abs(_sorted_affs(affs_n) - _sorted_affs(affs_p))) 
	affs_p, gr_p = affs_n, gr_n
    return affs_n, gr_n


def som1(patterns, gridshape=(10,1), alpha=0.99, r=2.0,
         neighbor_fn=neigh_gauss,
         fade_coeff = 0.9,
         min_reassign=10,
         max_iter = 1e5,
         distance=euclidean,
	 init_templates = None,
	 init_pca = False,
         random_query = True,
	 output = 'last',
         verbose = 0):
    """SOM as described in Bacao, Lobo and Painho, 2005
    Parameters:
      - `patterns` -- list or array-like, input patterns to train SOM against
      - `gridshape` -- shape of SOM grid
      - `alpha` -- \alpha parameter of SOM, "driving" force to adjust
         neighboring nodes
      - `r` -- radius for a neighbor function
      - `neighbor_fn` -- a function to define neighborhood
      - `fade_coeff` -- fading coefficient, parameters alpha and r are
         multiplied at each step
      - `min_reassign` -- stop after number of pattern reassignement has
         reached this value
      - `max_iter` -- don't do more than this number of iterations
      - `distance` -- distance function (Euclidean distance by default)
      - `init_templates` -- initialize SOM grid with this
      - `init_pca` -- if init_templates is None, initialize templates as
         first N principal components
      - `output` - string to define output, can be 'last', 'both' or 'full'
      - `verbose` - whether to be verboze 
 	 
    """

    if (isinstance(distance,str)) and distance in distance_fns:
        distance = distance_fns[distance]
    else:
        distance = euclidean

    ### TODOs:
    ### [X] go through patterns in random order, return sorted affiliations
    ### [X] use principal components as a start-off patterns
    ### [-] allow for sorting of at least 1D-grids
    niter = 0
    Npts = len(patterns)            # number of patterns
    sh = patterns[0].shape          # dimensionality
    grid = np.zeros(np.concatenate((gridshape, sh)))
    locs = list(itt.product(*map(xrange,gridshape)))
    L = len(locs)            # total dictionary size

    if init_templates is None:
	if init_pca:
	    patt = np.array([x.ravel() for x in patterns])
	    u,s,vh = np.linalg.svd(patt, full_matrices=False)
	    init_templates = [c.reshape(sh) for c in u[:len(locs)]]
	    del u,s,vh,patt
	else:
	    init_ks = np.random.randint(Npts, size=len(locs))
	    init_templates = [patterns[k] for k in init_ks]
    for k,l in enumerate(locs):
        grid[l] = init_templates[k]

    # initialize affiliations from patterns
    affiliations = np.array([classify(p, grid, distance)[0] for p in patterns])
    reassigned = len(affiliations)

    out = []
    grid_pdists = np.zeros((L,L))
    flatgrid_sh = (-1, ) + sh
    
    while alpha > 1e-6 and niter < max_iter and reassigned > min_reassign:
        affiliations_prev = affiliations.copy()
        for k1,k2 in itt.combinations(range(L),2):
            grid_pdists[k1,k2] = neighbor_fn(locs[k1],locs[k2],r)
            grid_pdists[k2,k1] = grid_pdists[k1,k2]
        for k,loc in enumerate(locs):
            grid_pdists[k,k] = neighbor_fn(loc,loc,r)
        ## Go through the patterns in random order
        if random_query:
            perm = np.random.permutation(Npts)
        else:
            perm = np.arange(Npts)

        for i,k in enumerate(perm):
            p = patterns[k]
            if (not k%100) and verbose:
                sys.stderr.write("\r %04d %06d, %06d"%(niter, reassigned, Npts-i))
            dists = distance(grid.reshape(flatgrid_sh), p)
            winner_ind = np.argmin(dists)
            affiliations[k] = winner_ind
            update = (p-grid.reshape(flatgrid_sh))
            aux_sh = (L,)+(1,)*len(sh)
            update *= grid_pdists[winner_ind].reshape(aux_sh)
            grid += alpha*update.reshape(grid.shape)
        alpha *= fade_coeff
        r *= fade_coeff
        reassigned = np.sum(affiliations != affiliations_prev)
        niter +=1
	out.append(affiliations.copy())
    if output == 'last':
	return affiliations
    elif output == 'grid':
	return grid
    elif output == 'both':
	return affiliations, grid
    elif output == 'full':
        return out, grid
    return out

def _som1_old(patterns, shape=(10,1), alpha=0.99, r=2.0, neighbor_fn=neigh_gauss,
              fade_coeff = 0.9,
              min_reassign=10,
              max_iter = 1e5,
              distance=euclidean,
              init_templates = None,
              init_pca = False,
              output = 'last',
              verbose = 0):
    """SOM as described in Bacao, Lobo and Painho, 2005
    Parameters:
      - `patterns` -- list or array-like, input patterns to train SOM against
      - `shape` -- shape of SOM grid
      - `alpha` -- \alpha parameter of SOM, "driving" force to adjust
         neighboring nodes
      - `r` -- radius for a neighbor function
      - `neighbor_fn` -- a function to define neighborhood
      - `fade_coeff` -- fading coefficient, parameters alpha and r are
         multiplied at each step
      - `min_reassign` -- stop after number of pattern reassignement has
         reached this value
      - `max_iter` -- don't do more than this number of iterations
      - `distance` -- distance function (Euclidean distance by default)
      - `init_templates` -- initialize SOM grid with this
      - `init_pca` -- if init_templates is None, initialize templates as
         first N principal components
      - `output` - string to define output, can be 'last', 'both' or 'full'
      - `verbose` - whether to be verboze 
 	 
    """

    if (isinstance(distance,str)) and distance_fns.has_key(distance):
        distance = distance_fns[distance]

    ### TODOs:
    ### [ ] go through patterns in random order, return sorted affiliations
    ### [X] use principal components as a start-off patterns
    ### [-] allow for sorting of at least 1D-grids
    niter = 0
    Npts = len(patterns)            # number of patterns
    L = len(patterns[0])            # dimensionality
    sh = patterns[0].shape          # dimensionality
    grid = np.zeros(np.concatenate((shape, sh)))
    locs = list(itt.product(*map(xrange,shape)))
    if init_templates is None:
	if init_pca:
	    patt = np.array([x.ravel() for x in patterns])
	    u,s,vh = np.linalg.svd(patt.T,full_matrices=False)
	    init_templates = [c.reshape(sh) for c in u.T[:len(locs)]]
	    del u,s,vh,patt
	else:
	    init_ks = np.random.randint(len(patterns), size=len(locs))
	    init_templates = [patterns[k] for k in init_ks]
    for k,l in enumerate(locs):
        grid[l] = init_templates[k]
    #affiliations = np.ones(Npts)*-1
    # initialize affiliations from patterns
    affiliations = np.array([_classify_old(p, grid, distance)[0] for p in patterns])
    reassigned = len(affiliations)
    out = []
    while alpha > 1e-6 and niter < max_iter and reassigned > min_reassign:
        affiliations_prev = affiliations.copy()
        for k,p in enumerate(patterns):
            if (not k%100) and verbose:
                sys.stderr.write("\r %04d %06d, %06d"%(niter, reassigned, Npts-k))
            dists = [distance(grid[loc],p) for loc in locs]
            winner_ind = np.argmin(dists)
            affiliations[k] = winner_ind
            winner_loc = locs[winner_ind]
            for loc in locs:
                grid[loc] += alpha*neighbor_fn(winner_loc, loc, r)*(p-grid[loc])
        alpha *= fade_coeff
        r *= fade_coeff
        reassigned = np.sum(affiliations != affiliations_prev)
        niter +=1
	out.append(affiliations.copy())
    if output == 'last':
	return affiliations
    elif output == 'grid':
	return grid
    elif output == 'both':
	return affiliations, grid
    return out

def classify(pattern, grid, distance = euclidean):
    sh = grid.shape[:2] # shape of grid
    locs = list(itt.product(*map(xrange,sh)))
    flatgrid_sh = (-1, ) + pattern.shape
    dists = distance(grid.reshape(flatgrid_sh), pattern)
    k = np.argmin(dists)
    return k, locs[k]


def _classify_old(pattern, grid, distance = euclidean):
    sh = grid.shape[:2] # shape of grid
    locs = list(itt.product(*map(xrange,sh)))
    dists = [distance(grid[loc],pattern) for loc in locs]
    k = np.argmin(dists)
    return k, locs[k]

def som_batch(patterns, shape=(10,1), neighbor_fn = neigh_gauss,
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
