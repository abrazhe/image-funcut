### Prototype of Growing neural gas

import numpy as np

class Node:
    def __init__(self, vector):
        self.edges = []
        self.vector = vector
        self.error = 0

class Edge:
    def __init__(self, nodes):
        self.nodes = nodes
        self.age = 0
