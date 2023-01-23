# -----------------------------------------------------------------------------
# Copyright (C) 2013-2022 Michael I. Love, Constantin Ahlmann-Eltze
# Copyright (C) 2023 Maximilien Colange

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
# -----------------------------------------------------------------------------

# This file is based on the file 'R/core.R' of the Bioconductor DESeq2 package
# (version 3.16).


import numpy as np

from ..utils import Factor


def collapseReplicates(obj, groupby, run, renameCols=True):
    """
    Collapse technical replicates in an AnnData or DESeqDataSet

    Collapse the samples (rows) in `obj` by summing within levels
    of a grouping factor `groupby`. The purpose of this function
    if to sum up read counts from technical replicates to create an object
    with a single row of read counts for each sample.
    Note: by "technical replicates", we mean multiple sequencing runs of the same
    library, in contrast to "biological replicates" in which multiple
    libraries are prepared from separate biological units.
    Optionally renames the columns of returned object with the levels of the
    grouping factor.

    Arguments
    ---------
    obj : AnnData or DESeqDataSet
    groupby : Factor
        a grouping factor, as long as the rows of obj
    run : ??, optional
        the names of each unique row in obj
        if provided, a new row "runsCollapsed" will be added to obj.obs
        which pastes together the names of `run`
    renameCols : bool, optional
        whether to rename the rows of the returned object using the levels
        of the grouping factor

    Returns
    -------
    AnnData or DESeqDataSet
        an object with as many rows as levels in `groupby`.
        This object has count data which is summed from the various rows which
        are grouped together, and its .obs is subset using the first row
        for each group in `groupby`.

    Examples
    --------
    >>> dds = makeExampleDESeqDataSet(m=12)
    >>> # make data with two technical replicates for three samples
    >>> dds.obs["sample"] = Factor(np.repeat(np.arange(1,10), [2,1,1,2,1,1,2,1,1]))
    >>> dds.obs["run"] = [f"run{i}" for i in range(12)]
    >>> ddsColl = collapseReplicates(dds, dds.obs["sample"], dds.obs["run"])
    >>> # examine the clinical data and rows names of the collapsed data
    >>> ddsColl.obs
    ???
    >>> ddsColl.index
    ???
    >>> # check that the sum of the counts for "sample0" is the same
    ... # as the counts in the "sample0" row in ddsColl
    >>> matchFirstLevel = dds.obs["sample"] == dds.obs["sample"].categories[0]
    >>> np.all(np.sum(dds[matchFirstLevel,:].counts(), axis=0) == ddsColl[0,:].counts())
    True
    """
    if not isinstance(groupby, Factor):
        groupby = Factor(groupby)
    groupby = groupby.droplevels()
    if len(groupby) != obj.n_obs:
        raise ValueError("groupby should have as many elements as observations in obj")

    countdata = [
        np.sum(obj.counts()[groupby == c, :], axis=0) for c in groupby.categories
    ]
    rowsToKeep = [np.nonzero(groupby == c)[0][0] for c in groupby.categories]
    collapsed = obj[rowsToKeep, :]
    collapsed.X = countdata
    if run is not None:
        if len(groupby) != len(run):
            raise ValueError("groupby and run should have the same length")
        collapsed.obs["runsCollapsed"] = [
            ",".join([f"{e}" for e in run[groupby == c]]) for c in groupby.categories
        ]

    if renameCols:
        collapsed.index = groupby.categories

    assert np.sum(obj.counts()) == np.sum(collapsed.counts())

    return collapsed
