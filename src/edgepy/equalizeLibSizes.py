import numpy as np
from .dropEmptyLevels import dropEmptyLevels
from .mglmOneGroup import mglmOneGroup
from .q2qnbinom import q2qnbinom

def equalizeLibSizes_DGEList(self, dispersion=None):
    # Check y
    y = validDGEList(self)

    # Check dispersion
    if dispersion is None:
        dispersion = y.getDispersion()

    lib_size = y.samples.lib_size * y.samples.norm_factors

    out = equalizeLibSizes(y.counts, group=y.samples.group, dispersion=dispersion, lib_size=lib_size)
    y.pseudo_counts = out.pseudo_counts
    y.pseudo_lib_size = out.pseudo_lib_size
    return y

def equalizeLibSizes(y, group=None, dispersion=None, lib_size=None):
    """
    Use a quantile-to-quantile transformation so that new counts are equivalent deviates on the equalized scale.
    """
    # Check y
    (ntags, nlibs) = y.shape

    # Check group
    if group is None:
        group = np.ones(nlibs)
    if len(group) != nlibs:
        raise ValueError("Incorrect length of group")
    group = dropEmptyLevels(group)

    # Check dispersion
    if dispersion is None:
        dispersion = 0.05

    # Check lib_size
    if lib_size is None:
        lib_size = y.sum(axis=0)
    if len(lib_size) != nlibs:
        raise ValueError("Incorrect length for lib_size")

    common_lib_size = np.exp(np.log(lib_size).mean())
    levs_group = group.levels
    input_mean = np.zeros(shape=(ntags, nlibs))
    output_mean = np.zeros(shape=(ntags, nlibs))
    for i in range(len(levs_group)):
        j = group.arr == levs_group[i]
        beta = mglmOneGroup(y[:,j], dispersion=dispersion, offset=np.log(lib_size[j]))
        lam = np.exp(beta)
        input_mean[:,j] = np.expand_dims(lam, 1) @ np.expand_dims(lib_size[j], 0)
        output_mean[:,j] = np.expand_dims(lam, 1) @ np.full((1, j.sum()), common_lib_size)
    pseudo = q2qnbinom(y, input_mean=input_mean, output_mean=output_mean, dispersion=dispersion)
    pseudo[pseudo<0] = 0
    return dict(pseudo_counts=pseudo, pseudo_lib_size=common_lib_size)
