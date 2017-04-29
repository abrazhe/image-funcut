"""
Handy decorators
"""

def with_time_dec(fn):
    "decorator to time function evaluation"
    def _(*args, **kwargs):
        import time
        t = time.time()
        out = fn(*args,**kwargs)
        print("time lapsed %03.3e in %s"%(time.time() - t, str(fn)))
        return out
    _.__doc__ = fn.__doc__
    return _

## not needed, use ipython magic %time
def with_time(fn, *args, **kwargs):
    "take a function and timer its evaluation"
    import time
    t = time.time()
    out = fn(*args,**kwargs)
    print("time lapsed %03.3e in %s"%(time.time() - t, str(fn)))
    return out
