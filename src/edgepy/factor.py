import numpy as np
from pandas import Categorical

class Factor(Categorical):
    def __init__(self, arr):
        super().__init__(arr)

#    def __len__(self):
#        return len(self.arr)
#
#    def __eq__(self, other):
#        if type(other) is type(self):
#            return np.array_equal(self.arr, other.arr)
#        else:
#            return NotImplemented

    def droplevels(self):
        """
        drop unused levels
        """
        return Factor(self.__array__())

    def nlevels(self):
        return len(self.categories)

def asfactor(g):
    if type(g) is Factor:
        return g
    else:
        return Factor(g)

def gl(n, k):
    arr = []
    for i in range(1, n+1):
        arr.extend([i for j in range(k)])
    return Factor(arr)

