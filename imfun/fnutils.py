### Functional programming utils for imfun package (moved from lib.py)

import sys

def fnchain(f,n):
    """
    returns lambda *args, **kwargs: f(..n times..f(*args, **kwargs))
    """
    return flcompose(*[f]*n)
	

def fniter(f,x,verbose=False):
    "Same as fnchain, but as an iterator"
    counter,out = 0,x
    while True:
	if verbose and not (counter%10):
	    sys.stderr.write('\r iteration %05d'%counter)
        out = f(out)
	counter += 1
	yield out


def take(N, seq):
    "Takes first N values from a sequence"	
    return [seq.next() for j in xrange(N)]


def flcompose2(f1,f2):
    "Compose two functions from left to right"
    def _(*args,**kwargs):
        return f2(f1(*args,**kwargs))
    return _
                  
def flcompose(*funcs):
    "Compose a list of functions from left to right"
    return reduce(flcompose2, funcs)
