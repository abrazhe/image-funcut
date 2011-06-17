# Optimization utilities

import numpy as np
from scipy import optimize
fmin = optimize.fmin
leastsq = optimize.leastsq

exp = np.exp

def gaussian(x, (a, b, c,f)):
    return a * np.exp(-(x-b)**2/(2*c**2))+f

def fwhm(c):
    return np.sqrt(2*np.log(2))*c

def score_gauss(p, x, v):
    return sum((v-gaussian(x,p))**2)




def gauss_fit(p0,x,y):
    return fmin(score_gauss, p0, (x, y))


def residuals_f(f):
    return lambda p, v, x: v - f(x,p)

def score_f(f):
    return lambda p, v, x: np.sum((v - f(x,p))**2)

def rising_exp(t, (a,b,tau)):
    return a - b*exp(-t/tau)

def expf(t, (a,b,tau)):
    return a + b*exp(-t/tau)

def double_rising_exp(t, (a,b1,b2,tau1,tau2)):
    return a - b1*exp(-t/tau1) - b2*exp(-t/tau2)

def logistic(t, (a,b,tau)):
    return a / (1 + b*exp(-t/tau))

def rising_pow(t, (a,b,alpha)):
    return a - b*t**(-alpha)

def mmenten(t, (a, km, alpha)):
    return a*t/(km + t**alpha)

def search_halfrise(tx, fn, p):
    maxv = np.max(fn(tx, p))
    searchfn = lambda a: abs(fn(a, p) - maxv/2.0)
    return fmin(searchfn, np.mean(tx))

def half_rise_mmenten(p):
    return p[1]**(1/p[2])

def half_rise_rising_exp(p):
    return p[2]*log(2*p[1]/p[0])

def half_rise_rising_pow(p):
    return p[2]*log(2*p[1]/p[0])



def iter_gauss(p0, vectors, vslice=slice(None)):
    out = []
    for v in vectors:
        x = np.arange(len(v))[vslice]
        p0 = gauss_fit(p0,x,v[vslice])
        out.append(p0)
    return out
