import itertools as itt
import numpy as np

def locations(shape):
    """ all locations for a shape; substitutes nested cycles
    """
    return itt.product(*map(xrange, shape))
    
def n_random_locs(n, shape):
    """
    return a list of n random locations within shape
    """
    return zip(*[tuple(np.random.choice(dim, n)) for dim in shape])


def make_grid(shape,size,stride):
    """Make a generator over sets of slices which go through the provided shape
       by a stride
    """
    origins =  itt.product(*[range(0,dim,stride) for dim in shape])
    squares = ([slice(a,a+size) for a in o] for o in origins)
    return squares

def in_circle(coords, radius):
    return lambda x,y: (square_distance((x,y), coords) <= radius**2)

def eu_dist(p1,p2):
    return np.sqrt(np.sum([(x-y)**2 for x,y in zip(p1,p2)]))

def eu_dist2d(p1,p2):
    "Euclidean distance between two points"
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def square_distance(p1,p2):
    return (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2
