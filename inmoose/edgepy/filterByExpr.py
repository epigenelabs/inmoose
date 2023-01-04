#-----------------------------------------------------------------------------
# Copyright (C) 2008-2022 Yunshun Chen, Aaron TL Lun, Davis J McCarthy, Matthew E Ritchie, Belinda Phipson, Yifang Hu, Xiaobei Zhou, Mark D Robinson, Gordon K Smyth
# Copyright (C) 2022-2023 Maximilien Colange

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#-----------------------------------------------------------------------------

# This file is based on the file 'R/filterByExpr.R' of the Bioconductor edgeR package (version 3.38.4).


def hat(x, intercept=True):
    #if intercept:
    #    x = # TODO cbind(1, x)
    n = x.shape[0]
    x = x.qr()
    #rowSums((Q D(1) )^2)

def filterByExpr(y, design=None, group=None, lib_size=None, min_count=10, min_total_count=15, large_n=10, min_prop=0.7):
    """
    Filter low expressed genes given count matrix
    Compute True/False index vector indicating which rows to keep
    """

    # TODO check that y is numeric

    if lib_size is None:
        lib_size = y.sum(axis=0)

    # Minimum effect sample size for any of the coefficients
    if group is None:
        if design is None:
            warn("No group or design set. Assuming all samples belong to one group.")
            MinSampleSize = y.shape[1]
        else:
            # TODO
            raise RuntimeError("Unimplemented case: 'group' is None while 'design' is not")
            h = hat(design)
            MinSampleSize = 1 / max(h)
    else:
        # TODO
        #group = as.factor(group)
        # TODO
        n = tabulate(group)
        MinSampleSize = min(n[n > 0])
    if MinSampleSize > large_n:
        MinSampleSize = large_n + (MinSampleSize - large_n) * min_prop

    # CPM cutoff
    MedianLibSize = median(lib_size)
    CPM_Cutoff = min_count / MedianLibSize * 1e6
    CPM = cpm(y, lib_size)
    tol = 1e-14
    keep_CPM = (CPM >= CPM_Cutoff).sum(axis=1) >= (MinSampleSize - tol)

    # Total count cutoff
    keep_TotalCount = y.sum(axis=1) >= (min_total_count - tol)

    return keep_CPM & keep_TotalCount

