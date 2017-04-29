import numpy as np
import scipy.interpolate as ip

def locextr(v, x=None, refine = True, output='full',
            sort_values = True,
            **kwargs):
       "Finds local extrema "
       if x is None: x = np.arange(len(v))
       tck = ip.splrep(x,v, **kwargs) # spline representation
       if refine:
               xfit = np.linspace(x[0],x[-1], len(x)*10)
       else:
               xfit = x
       yfit = ip.splev(xfit, tck)
       der1 = ip.splev(xfit, tck, der=1)
       #der2 = splev(xfit, tck, der=2)
       dersign = np.sign(der1)

       maxima = np.where(np.diff(dersign) < 0)[0]
       minima = np.where(np.diff(dersign) > 0)[0]
       if sort_values:
           maxima = sorted(maxima, key=lambda p: yfit[p], reverse=True)
           minima = sorted(minima, key=lambda p: yfit[p], reverse=False)
       if output=='full':
           return xfit, yfit, der1, maxima, minima 
       elif output=='max':
           return list(zip(xfit[maxima], yfit[maxima]))
       elif output =='min':
           return list(zip(xfit[minima], yfit[minima]))

def locextr_lsq_splines(v, x=None, points_per_knot=5,
                        refine=True, output='full',sort_values=True):
    if x is None: x = np.arange(len(v))
    t  = np.linspace(x[2],x[-2],len(v)/points_per_knot)
    spl = ip.LSQUnivariateSpline(x,v,t)
    if refine:
        xfit = np.linspace(x[0],x[-1], len(x)*10)
    else:
        xfit = x
    yfit = spl(xfit)
    der1 = spl.derivative()(xfit)
    dersign = np.sign(der1)

    maxima = np.where(np.diff(dersign) < 0)[0]
    minima = np.where(np.diff(dersign) > 0)[0]
    if sort_values:
        maxima = sorted(maxima, key=lambda p: yfit[p], reverse=True)
        minima = sorted(minima, key=lambda p: yfit[p], reverse=False)
    if output=='full':
        return xfit, yfit, der1, maxima, minima 
    elif output=='max':
        return list(zip(xfit[maxima], yfit[maxima]))
    elif output =='min':
        return list(zip(xfit[minima], yfit[minima]))

def extrema2(v, *args, **kwargs):
   "First and second order extrema"
   xfit,yfit,der1,maxima,minima = locextr_lsq_splines(v, *args, **kwargs)
   xfit, _, der2, gups, gdowns = locextr_lsq_splines(der1, x=xfit, refine=False)
   return (xfit, yfit), (maxima, minima), (gups, gdowns)
