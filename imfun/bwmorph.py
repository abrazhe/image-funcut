def contiguous_regions_2d(mask):
    """
    Given a binary 2d array, returns a sorted (by size) list of contiguous
    regions (True everywhere)
    TODO: make it possible to use user-defined funtion over an array instead of
    just a binary mask

    """
    import sys
    rl = sys.getrecursionlimit()
    sh = mask.shape
    sys.setrecursionlimit(sh[0]*sh[1])
    regions = []
    rows,cols = mask.shape
    visited = np.zeros(mask.shape, bool)
    for r in xrange(rows):
        for c in xrange(cols):
            if mask[r,c] and not visited[r,c]:
                reg,visited = cont_searcher((r,c), mask, visited)
                regions.append(reg)
    regions.sort(key = lambda x: len(x), reverse=True)

    sys.setrecursionlimit(rl)
    return map(lambda x: Region2D(x,mask.shape), regions)

def filter_proximity(mask, rad=3, size=5, fn = lambda m,i,j: m[i,j]):
    rows, cols = mask.shape
    X,Y = np.meshgrid(xrange(cols), xrange(rows))
    in_circle = lib.in_circle
    out = np.zeros((rows,cols), np.bool)
    for row in xrange(rows):
        for col in xrange(cols):
            if fn(mask,row,col):
                a = in_circle((col,row),rad)
                if np.sum(mask*a(X,Y))>size:
                    out[row,col] = True
    return out

def majority(mask, th = 5, mod = True):
    rows, cols = mask.shape
    out = np.zeros((rows,cols), np.bool)
    for row in xrange(rows):
        for col in xrange(cols):
            x = np.sum([mask[n] for n in neighbours((row,col),mask.shape)])
            out[(row,col)] = (x >= th)
            if mod:
               out[(row,col)] *= mask[row,col]
    return out
            

def filter_mask(mask, fn, args=()):
    """Split a mask into contiguous regions, filter their size,
    and return result as a mask
    """
    regs = contiguous_regions_2d(mask)
    filtered_regs = fn(regs, *args)
    z = np.zeros(mask.shape, dtype=np.bool)
    if len(filtered_regs) >1:
        return reduce(lambda a,b:a+b,
                      [z]+[r.tomask() for r in filtered_regs])
    else:
        return z

def filter_size_regions(regions, min_size=5):
    "Filters clusters by their size"
    return [r for r in regions if r.size()>min_size]

def filter_shape_regions(regions, th = 2):
    "Filters continuous regions by their shape"
    return [r for r in regions
            if (r.linsize() > th*np.sqrt(r.size()))]

def cont_searcher(loc, arr, visited):
    """
    Auxilary function for contiguous_regions_2d, finds one contiguous region
    starting from a non-False location
    TODO: make it possible to use user-defined function over an array instead of
    just a binary mask
    """
    acc = []
    def _loop(loc, acc):
        if visited[loc]:
            return
        visited[loc] = True
        if arr[loc] and (not loc in acc):
            acc.append(loc)
            for n in neighbours(loc, arr.shape):
                _loop(n,acc)
        else:
            return
    _loop(loc, acc)
    return acc, visited

## def cont_searcher_rec(loc,arr,visited):
##     if arr[loc] and (not loc in acc):
##         visited[loc]=True # side-effect!
##         return [loc] + [cont_searcher_rec(n) for n in neighbours(loc)
##                         if valid_loc(n,arr.shape)]
##     else:
##         return []


def neighbours(loc, shape):
    "list of adjacent locations"
    r,c = loc
    return filter(lambda x: valid_loc(x, shape),
                  [(r,c+1),(r,c-1),(r+1,c),(r-1,c),
                   (r-1,c-1), (r+1,c-1), (r-1, c+1), (r+1,c+1)])

def valid_loc(loc,shape):
    "location not outside bounds"
    r,c = loc
    return (0 <= r < shape[0]) and (0<= c < shape[1])

class Region2D:
    "Basic class for a contiguous region. Can make masks from it"
    def __init__(self, locs, shape):
        self.locs = locs
        self.shape = shape
    def size(self,):
        return len(self.locs)
    def center(self):
        return np.mean(self.locs,0)
    def linsize(self,):
        dists = [lib.eu_dist(*pair) for pair in itt.permutations(self.locs,2)]
        return reduce(max, dists)
                               
        pass
    def tomask(self):
        m = np.zeros(self.shape, bool)
        for i,j in self.locs: m[i,j]=True
        return m
            
