from .dropEmptyLevels import dropEmptyLevels
import numpy as np

def splitIntoGroups(self):
    group = self.samples.group
    return splitIntoGroups(self.counts, group=group)

def splitIntoGroups(y, group=None):
    # Check y
    (ntags, nlibs) = y.shape

    # Check group
    if group is None:
        group = np.ones(nlibs)
    if len(group) != nlibs:
        raise ValueError("Incorrect length of group.")
    group = dropEmptyLevels(group)

    out = [y.T[group == i].T for i in np.unique(group)]
    return out
