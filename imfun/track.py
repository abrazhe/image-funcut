import lib
import numpy as np
import scipy.interpolate as ip


import atrous

def limit_bounds(vec, lbound, ubound):
   out = np.copy(vec)
   if lbound:
      out = out[out >= lbound]
   if ubound:
        out = out[out <= ubound]
   return out

def gaussian(mu, sigma):
    return lambda _x: np.exp(-(_x-mu)**2/(2*sigma**2))


def track1(seq, seed, memlen = 5, 
           guide = None, 
           lbound = None, ubound = None):
    out = []
    seeds = seed*np.ones(memlen)
    use_guide = (guide is not None and len(guide) >= len(seq))
    for k,el in enumerate(seq):
        el = limit_bounds(el, lbound, ubound)
        if len(el) > 0:
            if use_guide: 
                target = (np.sum(seeds) + guide[k])/(len(seeds)+1)
            else:
                target = np.mean(seeds)
            j = np.argmin(abs(el - target))
            seeds[:-1] = seeds[1:]
            seeds[-1] = el[j]
            out.append((k,el[j]))
    return np.array(out)

## def spl1st_derivative(ck):
##     L = len(ck)
##     return [0.5*(ck[mirrorpd(k+1,L)] - ck[mirrorpd(k-1,L)])  for k in range(L)]

## def spl2nd_derivative(ck):
##     L = len(ck)
##     return np.array([ck[mirrorpd(k+1,L)] + ck[mirrorpd(k-1,L)] - 2*ck[k] for k in range(L)])

def mirrorpd(k, L):
    if 0 <= k < L : return k
    else: return -(k+1)%L


def locextr(v, x=None, mode = 'max', refine=10, output='xfit'):
   """Finds local extrema, type of extrema depends on parameter "mode"
   mode can be {'max' | 'min' | 'gup' | 'gdown' | 'gany'}"""

   if type(x) is str:
       mode = x

   if x is None or type(x) is str:
       x = np.arange(len(v))
       
   sp0 = ip.UnivariateSpline(x,atrous.smooth(v),s=0)
   if mode in ['max', 'min']:
       sp = sp0.derivative(1)
   elif mode in ['gup', 'gdown', 'gany']:
       sp = sp0.derivative(2)
   res = 0.05
   if refine > 1:
       xfit = np.linspace(0,x[-1], len(x)*refine)
   else:
       xfit = x
   di = sp(xfit)
   if mode in ['max', 'gup']:
       dersign = np.sign(di)
   elif mode in ['min', 'gdown']:
       dersign = -np.sign(di)
   locations = dersign[:-1] - dersign[1:] > 1.5
   
   if output is 'all':
       out =  xfit[locations], sp0(xfit)[locations]
   elif output is 'yfit':
       out = di[locations]
   elif output is 'xfit':
       out = xfit[locations]
   else:
       print """unknown output code, should be one of  'xfit', 'yfit', 'all',
       returning 'x' locations"""
       out = xfit[locations]
   return out

def guess_seeds(seq, Nfirst=10,smoothing=4):
    """
    automatically guess starting seeds for vessel walls
    relies on the fact that vessel is the largest bright stripe
    """
    Nfirst = min(len(seq), Nfirst)
    y = atrous.smooth(np.mean(seq[:Nfirst],axis=0), smoothing)
    (xfit,yfit), (mx,mn), (gups,gdowns) = lib.extrema2(y, sort_values=True)
    # highest gradient up to the left of the highest max
    gu1 = (g for g in gups if g < mx[0]).next()
    # highest gradient up to the right of the highest max
    gd1 = (g for g in gdowns if g > mx[0]).next() 
    return (xfit[gu1], xfit[gd1])


def follow_extrema(arr, start, mode='gany', memlen=5):
    """
    modes: {'gup' | 'gdown' | 'min' | 'max' | 'gany'}
    ##(gany -- any gradient extremum, i.e. d2v/d2x =0)
    """
    def _ext(v):
        ##(xf,yf),(mx,mn), (gups, gdowns) = lib.extrema2(v)
        if mode in ['gup', 'gdown', 'max', 'min']:
            return locextr(v, mode=mode)
        elif mode =='gany':
            return np.concatenate([locextr(v,mode='gup'),locextr(v,mode='gdown')])
    extrema = map(_ext, arr)
    v = track1(extrema, start, memlen=memlen)
    return v


def track_pkalman(xv,seq, seed_mu,seed_var=4.,gain=0.25):
    history = seed_mu #seed_mu*np.ones(memlen)
    delta=0
    out = []
    for el in seq:
        mu = history
        # broadness of window depends on previous scatter in positions and
        # inversly on SNR in the current frame (within a window around
        # predicted position
        snr = np.max(el)/np.std(el)
        var = seed_var + delta**2 + (1./snr)**2
        p = gaussian(mu,var**0.5)(xv)*el
        z = xv[np.argmax(p)]
        g = (1+snr)/(1+snr/gain)
        x = g*mu + (1-g)*z
        history = x
        delta = mu-x
        out.append(x)
    return np.array(out)
    
def v2grads(v):
    L = len(v)
    sp2 = ip.UnivariateSpline(np.arange(L),atrous.smooth(v), s=0)
    xfit = np.arange(0,L,0.1)
    return np.abs(sp2.derivative(1)(xfit))

def track_walls(linescan,output = 'kalman',Nfirst=None,gain=0.25):
    '''
    output can be one of ('kalman',  'extr', 'mean', 'all')
    '''
    if Nfirst is None: Nfirst = len(linescan)/2
    seeds = guess_seeds(linescan,Nfirst)
    xfit = np.arange(0,linescan.shape[1],0.1)
    grads = np.array(map(v2grads, linescan))
    if output in ['kalman', 'mean', 'all']:
        tk1,tk2 = [track_pkalman(xfit, grads,seed,gain=gain) for seed in seeds]
    if output in ['extr', 'mean', 'all']:
        te1,te2 = [follow_extrema(linescan, seed)[:,1] for seed in seeds] 
    if output == 'mean':
        return 0.5*(tk1+te1), 0.5*(tk2+te2)
    elif output == 'kalman':
        return tk1,tk2
    elif output == 'extr':
        return te1,te2
    elif output == 'all':
        return tk1,tk2, te1,te2

### ------------------------------------------
### --  Active contours  as particle system --
### ------------------------------------------


## [ ] TODO: use adaptive time-stepping scheme
## [ ] TODO: update fast-changing equations with finer time-steps 

class LCV_Contours:
    """
    Vertically-constrained Chan-Vese-ispired active contours as
    Verlet-integrated connected-particle system
    """
    def __init__(self, (low_contour, upper_contour),
                 U,
                 stiffness=0.25,
                 damping=0.25,
                 lam = 1.0,
                 thresh=0.,
                 max_force=2.):
    
        L = np.min((U.shape[1],len(low_contour),len(upper_contour)))
        self.L = L
        self.U = U
        self.upper_bound = U.shape[0]-1
        self.thresh = thresh
        self.set_weights()
        
        self.conts = np.concatenate((low_contour[:L], upper_contour[:L]))
        self.check_conts()
        self.contprev = self.conts.copy()     
        self.acc = np.zeros(self.conts.shape)
    
        self.lam = lam
        
        self.damping = damping
        self.hooke = stiffness
        self.max_force = max_force
        self.xind = np.arange(L)
        self.yind = np.arange(self.U.shape[0])
        self.dt = 1./self.max_force
        self.niter = 0
    
    def set_weights(self):
        snrv = lib.simple_snr2(self.U)
        self.weights = np.float_(snrv > self.thresh)
        #self.weights[:50] = 0
    
    def check_conts(self):
        """ensure vessel conts are within boundaries and ordered as low,upper"""
        w = self.conts.reshape(2,-1)
        w = np.where(w < 0, 0, w)
        w = np.where(w > self.upper_bound, self.upper_bound, w)
        self.conts = np.sort(w, 0).reshape(-1)
    
    def get_diameter(self):
        self.check_conts()
        l,u = self.conts.reshape(2,-1)
        return u-l
    def get_lower(self):
        return self.conts[:self.L]
    def get_upper(self):
        return self.conts[self.L:]
        
    def update_accelerations(self):
        xind = self.xind
        L = self.L
        self.check_conts()
        
        lower, upper = self.conts.reshape(2,-1)

        a = np.zeros_like(self.acc)
        #a -= self.damping*(self.conts-self.contprev)  # friction
        
        a[1:L] -= self.hooke*(lower[1:]-lower[:-1]) # smoothing forces
        a[:L-1] -= self.hooke*(lower[:-1]-lower[1:]) # fixme with signs
        
        a[L+1:] -= self.hooke*(upper[1:]-upper[:-1]) # smoothing terms
        a[L:-1] -= self.hooke*(upper[:-1]-upper[1:]) # fixme with signs
        
        a[:L] -= self.hooke*(self.weights==0)*(lower-np.mean(lower)) # where weights are zero, 
        a[L:] -= self.hooke*(self.weights==0)*(upper-np.mean(upper)) # push to the mean position
        
        rc = self.yind
        
        for j in xind[:]:
            l,u = lower[j],upper[j]
            inside = (rc > l) * (rc < u)
            outside = -inside # outside is a negation of inside
            c1 = np.sum(self.U[inside,j])/np.float(np.sum(inside+1e-12))
            c2 = np.sum(self.U[outside,j])/np.float(np.sum(outside+1e-12))
            v = np.abs(self.U[:,j]-c1) - np.abs(self.U[:,j]-c2)
            vmax = abs(v).max()
            a[j] += self.lam*v[l]*self.weights[j]/vmax
            a[L+j] -= self.lam*v[u]*self.weights[j]/vmax
        self.acc = a
        self.acc = np.sign(a)*np.where(abs(a)>self.max_force, self.max_force, abs(a)) # acceleration limit
        return a
    
    def verlet_step(self):
        dt = 0.5  
        ## use .update_accelerations before .verlet_step
        ynext = (2-self.damping)*self.conts - \
                (1-self.damping)*self.contprev +\
                self.acc*dt**2
        ynext = np.where(ynext < 0, 0, ynext)
        ynext = np.where(ynext > self.upper_bound, self.upper_bound, ynext)
        
        self.contprev = self.conts
        self.conts = ynext      
        self.niter += 1
        return self.conts


def solve_contours_animated(lcvconts, niter=500,
                            tol=0.001,
                            skipframes=5,
                            kstop = 15):
    import matplotlib.pyplot as plt
    f,a = plt.subplots(1,1)
    a.imshow(lcvconts.U, cmap='gray', interpolation='nearest', aspect='auto')
    lh0 = [a.plot(lcvconts.xind, w, 'g-')[0] for w in lcvconts.conts.reshape(2,-1)] # starting points
    lh = [a.plot(lcvconts.xind, w, color='orange')[0] for w in lcvconts.conts.reshape(2,-1)]
    a.axis('tight')
    names = []
    acc = []
    whist = []
    prevd = lcvconts.get_diameter()
    k = 0
    errchange = 1+tol
    L = lcvconts.L
    for i in xrange(niter):
        lcvconts.update_accelerations()
        lcvconts.verlet_step()
        d = lcvconts.get_diameter()
        err = np.mean(abs(d - prevd)) # todo: better stopping condition
        prevd = d
        acc.append(err)
        f.canvas.draw()
        if (i+1)%skipframes == 0:
            lh[0].set_ydata(lcvconts.conts[:L])
            lh[1].set_ydata(lcvconts.conts[L:])
        if i>kstop:
            errchange = np.polyfit(np.arange(kstop),acc[-kstop:],1)[0]
        if i>kstop and (err < tol or (np.abs(errchange)<tol)):
            if k >= kstop:
                # make it be for k times that (e.g. k=10)
                print 'Converged in %d iterations'%(i+1)
                break
            else:
                k+=1

    lh[0].set_ydata(lcvconts.conts[:L]) # last graph update
    lh[1].set_ydata(lcvconts.conts[L:]) #
    lh[0].set_color('r') # red is for final contour
    lh[1].set_color('r')
    plt.draw()
    return lh
