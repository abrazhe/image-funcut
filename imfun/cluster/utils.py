import numpy as np
import itertools as itt

def _sorted_memberships(membs):
    return np.array(sorted(list(range(int(np.max(membs)+1))), key=lambda i:np.sum(membs==i)))

def sort_clusters_by_size(membs):
    """
    Sort cluster indices such that cluster with smallest index has 
    most members.
    """
    sorted_idx = _sorted_memberships(membs)[::-1]
    new_membs = np.zeros_like(membs)
    for k,idx in enumerate(sorted_idx):
        new_membs[membs==idx] = k
    return new_membs




def select_points(points, memberships, idx):
    return [p for a,p in zip(memberships, points) if a == idx]


def filter_clusters_size(clusters, min_size=100):
    return [x for x in clusters if x.mass() > min_size]

def plot_clusters(points, clusters):
    import pylab as pl
    pl.figure(figsize=(6,6))
    arr = points2array(points)[:,:2]
    pl.scatter(*arr.T[:2,:], color='k', s=1)
    colors = ['r','b','g','c','m','y']
    for j,c in enumerate(clusters):
        pl.scatter(*cluster2array(c).T[:2,:], color=colors[j%len(colors)],
                alpha=0.5)

def plot3_clusters(points, clusters):
    
    from mpl_toolkits.mplot3d import axes3d

    pl.figure(figsize=(6,6))
    ax = pl.axes(projection='3d')
    arr = points2array(points)[:,:3]
    ax.plot(*arr.T[:3,:], color='k', ls='none',
         marker=',', alpha=0.3)
    colors = ['r','b','g','c','m','y']
    for j,c in enumerate(clusters):
        ax.plot(*cluster2array(c).T[:3,:],
             ls = 'none', marker=',',
             color=colors[j%len(colors)],
             alpha=0.5)


def cluster2array(c):
    "helpful for scatter plots"
    return points2array(c.points)

def points2array(points,dtype=np.float64):
    return np.array(points, dtype=dtype)


