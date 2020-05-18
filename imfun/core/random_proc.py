import numpy as np


def ar1(alpha=0.74):
    "Simple auto-regression model"
    randn = np.random.randn
    prev = randn()
    while True:
        res = prev * alpha + randn()
        prev = res
        yield res


from collections import deque


def ar_process(noise, p):
    buff = deque()
    memlen = len(p)
    for i, n in enumerate(noise):
        upd = n + np.sum(phi * b for phi, b in zip(p, buff))
        buff.appendleft(upd)
        if len(buff) > memlen:
            buff.pop()
            yield upd


def arma_process(noise, phiv, thetav, c=0):
    """
    Create generator for an ARMA random process
    Inputs:
      - noise : (\epsilon ~ N) sequence to filter
      - phiv  : \phi coefficients of the AR component
      - thetav: \theta coefficients of the MA component
    """
    p, q = len(phiv), len(thetav)
    buff_ar, buff_ma = deque(), deque()
    for i, n in enumerate(noise):
        upd = n
        upd += np.sum(phi * b for phi, b in zip(phiv, buff_ar))
        upd += np.sum(th * b for th, b in zip(thetav, buff_ma))
        buff_ar.appendleft(upd)
        buff_ma.appendleft(n)
        if len(buff_ma) > q:
            buff_ma.pop()
        if len(buff_ar) > p:
            buff_ar.pop()
        if i > p and i > q:
            yield (upd + c)
