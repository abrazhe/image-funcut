### Small lib to create synthetic data.
### Look at random_spikes() and synth_movie()

import itertools as itt
import numpy as np
from numpy import pi, cos, sin, tan

def spike(x, location, a, tau, rise=20):
    "a single spike"
    coef = 0.5 * a * (1 + np.tanh(rise * (x-location) / tau))
    return coef * np.exp(- abs(x -location)/ tau)


def randn_iter(loc, sigma):
    while True :
        yield np.random.normal(loc, sigma)

def ar1(loc, sigma, alpha=0.8):
    rnit = randn_iter(0, sigma)
    x = rnit.next()
    while True:
        yield loc+x
        x = rnit.next() + alpha*x
    
def rand_arrivals(interval, sigma = None, rand_process = ar1):
    if sigma is None:
        sigma = 0.15*interval # 15 % of mean
    #rnit = randn_iter(interval, sigma)
    phase_shift = rand_process(interval, 0.5*interval).next()
    rnit = rand_process(interval, sigma)
    x = phase_shift
    while True:
        yield x
        x += rnit.next()


def random_spikes(tvec, isi=1.2, amp=10, tau=0.15,
                  sigma = None, spikef= None,
                  snr = 2):
    rl = rand_arrivals(isi, sigma)
    max_t = tvec[-1]
    locations = itt.takewhile(lambda x:x<max_t, rl)
    out = np.add.reduce([spike(tvec, loc, amp, tau) for loc in locations])
    if snr > 0:
        noise_sigma = np.std(out)*np.sqrt(1.0/snr)
        out += np.random.normal(0,noise_sigma,len(tvec))
    return out

def regular_spikes(tvec, stim_freq, (stim_start, stim_stop), amp=10, tau=0.15, rise=20):
    locations = np.arange(stim_start, stim_stop, 1./stim_freq)
    out = np.add.reduce([spike(tvec, loc, amp, tau,rise) for loc in locations])
    return out

_pi = 3.14159265359

def synth_movie(tvec, size, nobjs=42, snr=0.5,
                spike_pars = {},
                dendr_length =64, angle=_pi/5):
    L = len(tvec)
    spike_pars_use = {'isi':1.2, 'amp':200, 'snr':100}
    spike_pars_use.update(spike_pars)
    print spike_pars_use
    masks = dendrite_masks(nobjs, dendr_length, angle)
    spike_signals = [random_spikes(tvec,**spike_pars_use) for i in xrange(nobjs)]
    signal_mean = np.mean(map(np.mean, spike_signals))
    signal_std = np.mean(map(np.std, spike_signals))
    noise = lambda : np.random.poisson((signal_std**2)/snr, size=L)
    out = np.zeros((L, size, size))
    for j in xrange(size):
        for k in xrange(size):
            out[:,j,k] = noise()
            for n,mask in enumerate(masks):
                if mask[j,k]:
                    out[:,j,k] +=spike_signals[n]
              
    return out

def synth_movie_from_seq(seq, baseL, type=1):
    d1 = seq.as3darray()
    L = seq.length()
    sdf = np.std(d1[:baseL,:,:], axis=0)
    mf = np.mean(d1[:baseL,:,:], axis=0)
    out = np.zeros(d1.shape)
    if type==1: lamm = mf
    else: lamm=sdf
    for s,j,k in seq.pix_iter():
        out[:,j,k] = np.random.poisson(lamm[j,k], size=L)
    return out

def ellipse(xc, yc, a, b, phi):
    t = arange(0, 2*pi, 0.01)
    X = xc + a* cos(t) * cos(phi) + b *sin(t) * sin(phi)
    Y = yc - a* cos(t) * cos(phi) + b *sin(t) * sin(phi)
    return X, Y


def eu_dist(p1,p2):
    "Euler distance between two points"
    return sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def dendrite_masks(nmasks, mlength, mangle, sigma=5, width=0.5, size=128):
    urand = lambda : np.random.uniform(mlength/10, size-mlength)
    X,Y = np.meshgrid(range(size),range(size))
    out = []
    k,n,max_tries = 0,0,1e4
    while k < max_tries:
        l = LineSegment((urand(), urand()),
                        np.random.normal(mlength, sigma),
                        mangle)
        m = l.mask1((X,Y), width)
        if n < 1:
            out.append(m)
            occupied = m
            n+=1
        else:
            overlap = np.sum(m*occupied)
            if not overlap:
                out.append(m)
                occupied += m
                n +=1
        if n > nmasks: break
    return out[1:]


def gauss2d(xmu=0, ymu=0, xsigma=10, ysigma=10):
    xsigma, ysigma = map(np.float, [xsigma, ysigma])
    return lambda x,y: np.exp(-(x-xmu)**2/(2*xsigma**2) - (y-ymu)**2/(2*ysigma**2))

def gauss3d(tscale=10, xscale=20, yscale=20):
    tscale, xscale, yscale = map(np.float, [tscale, xscale, yscale])
    t,x,y = np.mgrid[-tscale:tscale, -yscale:yscale, -xscale:xscale]
    return np.exp(-t**2/tscale - y**2/xscale - x**2/yscale)


def simple_wave(sstart = 1.0, alpha=0.8, shape=(128,128)):
    origin = (63.,63.)
    sigma = sstart
    amp = sstart
    count = 0
    for k in xrange(50):
	fn = gauss2d(origin[0], origin[1], sigma,sigma)
	def _(x,y):
	    print amp, sigma, 1/sigma**0.25
	    return np.tanh(10*amp*fn(x,y))
	yield _
	sigma += sqrt(alpha*k/sigma)
	amp -= 0.05*sigma**0.25
	if amp < 0.05:
	    amp = 0
	
	

def _____waves(shape, nobj=5):
    #out = np.zeros(shape, np.float32)
    xs,ys = shape[1:]
    objs = [gauss3d(*np.random.poisson(40,size=3)) for i in range(nobj)]
    tlocs = np.random.uniform(20, shape[0]-20, size=nobjs)
    xlocs = np.random.uniform(20, xs-20, size=nobjs)
    ylocs = np.random.uniform(20, ys-20, size=nobjs)
    tmax = objs[-1].shape[0] + tlocs[-1] + 5
    

class LineSegment:
    def __init__(self, p0, length, phi):
        self.p0 = map(float, p0)
        self.p1 = [p0[0] + length*cos(phi), p0[1] + length*sin(phi)]
        self.length = length
        self.phi = phi
        self.k = tan(phi)
        self.n = p0[1] - p0[0]*self.k
    def in_rect(self, point):
        x,y = point
        x1,x2 = self.p0[0], self.p1[0]
        y1,y2 = self.p0[1], self.p1[1]
        out =  (x < max(x1,x2)) * (x > min(x1,x2))
        out *= (y < max(y1,y2)) * (y > min(y1,y2))
        return out

    def dist_from_line(self, point):
        x,y  = point
        asq = (x - (y-self.n)/self.k)**2
        bsq = (y - (self.k*x + self.n))**2
        return (asq*bsq)/(asq+bsq)
    def mask1(self,point, dist):
        return self.in_rect(point) * (self.dist_from_line(point) < dist)
    def mask2(self, point, dist):
        d1 = eu_dist(self.p0, point)
        d2 = eu_dist(self.p1, point)
        return (d1 + d2) < self.length + dist
    
    

