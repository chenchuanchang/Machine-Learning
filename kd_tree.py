import numpy as np
class node:
    def __init__(self, value, axis, rb, lb):
        """
        Param
        value: node value [M]
        axis: split axis
        rb: right pointer
        lb: left pointer

        Node of kd-tree
        """
        self.value = value
        self.axis = axis
        self.rb = rb
        self.lb = lb

class kd_tree:
    def __init__(self, X):
        """
        Input
        X: data [N, M]

        Building a kd-tree
        """
        self.head = self.build(X)

    def build(self, X):
        pass

    def search(self, x, k):
        """
        Input
        x: target data [M]
        k: number of nearest neighbor

        Output
        y: k nearest neighbor List[k]

        Search k nearest neighbors
        """

        pass