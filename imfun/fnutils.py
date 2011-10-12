### Functional programming utils for imfun package (moved from lib.py)


def fnchain(f,n):
    """
    returns lambda *args, **kwargs: f(..n times..f(*args, **kwargs))
    """
    return flcompose(*[f]*n)
	

def fniter(f,x):
    "Same as fnchain, but as an iterator"
    out = x
    while True:
        out = f(out)
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
