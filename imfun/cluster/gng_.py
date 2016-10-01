### Prototype of Growing neural gas

import numpy as np
import random as pyrnd

from pylab import *

from .metrics import euclidean, cityblock, pearson, spearman, xcorrdist

class Edge:
    def __init__(self, node1, node2):
        self.nodes = (node1,node2)
        self.age = 0
    def length(self, distfn=euclidean):
        n1, n2 = self.nodes
        return distfn(n1.vector,n2.vector)
    def __repr__(self):
        return "Edge: " + str(self.nodes)


class BasicNode:
    def __init__(self, vector=None):
        self.edges = []
        self.vector = np.array(vector)
    def move(self, shift):
        self.vector =  self.vector+shift
    def neighbor(self, edge):
        n1,n2 = edge.nodes[:2]
        if n1 is self: return n2
        else: return n1
    def allneighbors(self):
        return [self.neighbor(e) for e in self.edges]
    def connected(self, node):
        for e in self.edges:
            if node in e.nodes:
                return e
        return False



class GNGNode(BasicNode):
    def __init__(self, vector=None):
        self.edges = []
        self.vector = np.array(vector)
        self.error = 0
    def __repr__(self):
        return str(self.vector)
    def connect(self, n):
        newedge = Edge(self, n)
        self.edges.append(newedge)
        n.edges.append(newedge)
        return newedge
    def pull_neighbors(self, en, ea=0.1):
        for e in self.edges:
            n = self.neighbor(e)
            shift = en*(self.vector - n.vector)
            n.move(shift)
            e.age += 1 + ea*e.length() # make longer edges age a bit faster


def connect_nodes(n1,n2):
    newedge = Edge(n1, n2)
    n1.edges.append(newedge)
    n2.edges.append(newedge)
    return newedge

def disrupt_edge(n1,n2):
    connection = n1.connected(n2)
    n1.edges = [e for e in n1.edges if e != connection]
    n2.edges = [e for e in n2.edges if e != connection]

def plot_graph(nodes, color='m', max_age=500, marker = 'o'):
    edges = []
    for n in nodes:
        x,y = n.vector[:2]
        if hasattr(n, 'age'):
            alpha = 1/(float(n.age)/max_age + 1.0)
        else:
            alpha = 1
        plot(x,y,ls='none',marker=marker,color=color, alpha=alpha)
        for e in n.edges:
            if not e in edges:
                v1,v2 = [n.vector for n in e.nodes]
                plot([v1[0], v2[0]], [v1[1], v2[1]], '-', color=color,
                     alpha = 1/(e.age/max_age+1.0))
                edges.append(e)
    return

def remove_stale_edges(nodes, max_age):
    for n in nodes:
        for e in n.edges: e.age+=0.25
        n.edges = [e for e in n.edges if e.age < max_age]

def remove_lonely_nodes(nodes):
    return [n for n in nodes if len(n.edges)>0]

def gng_add_node(nodes, alpha):
    keyfn = lambda _n: _n.error
    u = sorted(nodes, key=keyfn)[-1]
    y = sorted(u.allneighbors(), key=keyfn)[-1]
    newnode = GNGNode(0.5*(y.vector + u.vector))
    connect_nodes(u,newnode)
    connect_nodes(y,newnode)
    disrupt_edge(u,y)
    u.error *= alpha
    y.error *= alpha
    newnode.error = u.error
    nodes.append(newnode)
    return nodes

def _gngrun(data, ew = 0.1, en = 0.001, _lambda = 100, d = 0.9,
            max_age = 500, max_iter = 1e6,
            nodes = None, alpha = 0.5, beta = 0.0005,
            extent = None,
            animate = False,
            distfn = euclidean,
            max_nodes = 512):
    if nodes is None:
        nodes = map(GNGNode, pyrnd.sample(data, 2))
        edges = [connect_nodes(*nodes)]
    k,L = 1,len(data)
    while (k < max_iter) and (len(nodes) < max_nodes):
        #x = pyrnd.sample(data,1)[0]
        x = data[(k-1)%L]
        keyfn = lambda _n: distfn(_n.vector, x)
        ws,wt = sorted(nodes, key = keyfn)[:2]
        ws.error += distfn(ws.vector,x)**2
        shift =  ew*(x-ws.vector)
        ws.move(shift)
        ws.pull_neighbors(en)
        e = ws.connected(wt)
        if e: e.age = 0
        else: connect_nodes(ws, wt)
        remove_stale_edges(nodes, max_age)
        nodes = remove_lonely_nodes(nodes)
        for node in nodes: node.error *= d
        if not (k%_lambda):
            nodes = gng_add_node(nodes, alpha)
            if animate:
                cla()
                plot_graph(nodes, 'g', max_age = max_age, marker=',')
                axis(extent)
                draw()
        for n in nodes:
            n.error -= n.error*beta
        k+=1
    return nodes

def coefs2points(coefs):
    points = []
    nr,nc = coefs.shape[1:]
    for r in xrange(nr):
        for c in xrange(nc):
            v = coefs[:,r,c]
            loc = [float(c)/nc, float(r)/nr]
            points.append(np.concatenate([loc, v]))
    return points

### ---------- Particle search ------------------

## What I almost missed here was collective behaviour
            
class Particle(BasicNode):
    def __init__(self, vector):
        BasicNode.__init__(self,vector)
        self.k = 0.001
        self.age = 0
        self.D = 10.0
    def rand_move(self, surface):
        nr,nc = surface.shape
        u = np.random.uniform
        #for e in self.edges:
        #    n = self.neighbor(e)
        #    d = euclidean(n.vector, self.vector)
        #    n.move(self.k*shift/d)
        #    #shift += self.k*
        c,r = self.vector
        p =  u(surface.min(), surface.max())
        if  p > surface[c,r]:
            shift = np.round(u(-1,1)), np.round(u(-1,1))
            self.vector += shift
            self.vector[0] = self.vector[0] % (nc-1)
            self.vector[1] = self.vector[1] % (nr-1)
    def ageinc(self):
        self.age += 1.0/(1.0 + 100.0*len(self.edges)) 

        



def rand_pos(maxx,maxy):
    u = np.random.uniform
    return map(int, map(np.round, (u(0,maxx-1),u(0,maxy-1))))

def connect_close_particles(particles, min_dist = 3):
    for p in particles:
        for p2 in particles:
            if (p2 is not p) and \
                   not p.connected(p2) and \
                   (euclidean(p.vector, p2.vector) < min_dist):
                    connect_nodes(p,p2)
    return particles
def disrupt_long_connections(particles, max_dist = 10):
    for p in particles:
        for e in p.edges:
            n1,n2 = e.nodes
            if euclidean(n1.vector, n2.vector) > max_dist:
                disrupt_edge(*e.nodes)
    return particles
def remove_old_particles(particles, max_age):
    return [p for p in particles if p.age < max_age]
        
def particle_search(surface, add_interval = 100,
                    particles = None,
                    min_dist = 4,
                    max_dist = 10,
                    animate = False,
                    extent = None,
                    max_particles = 200,
                    max_age = 500,
                    maxiter = 1e6):
    k = 1
    nr,nc = surface.shape
    if particles is None:
        particles = [Particle(rand_pos(nc,nr))]
    while k <= maxiter and len(particles) < max_particles:
        if not k%add_interval:
            particles.append(Particle(rand_pos(nc,nr)))
        for p in particles:
            p.rand_move(surface)
            p.ageinc()
        connect_close_particles(particles)
        disrupt_long_connections(particles)
        L=len(particles)
        particles = remove_old_particles(particles, max_age)
        if len(particles) < L:
            particles.append(Particle(rand_pos(nc,nr)))
        if animate:
            cla()
            plot_graph(particles, 'g', max_age = max_age, marker=',')
            axis(extent)
            draw()
        k+=1
    return particles
        
    
