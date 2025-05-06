# -----------------------------------------------------------------------------
# Copyright (C) 2013-2022 Michael I. Love, Constantin Ahlmann-Eltze
# Copyright (C) 2023-2024 Maximilien Colange

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

# This file is based on the file 'R/plot.R' of the Bioconductor DESeq2 package
# (version 3.16).


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plotMA_res(
    self,
    alpha=None,
    main="",
    xlab="mean of normalized counts",
    ylab="log fold change",
    ylim=None,
    colNonSig="grey",
    colSig="blue",
    colLine="grey",
    returnData=False,
    MLE=False,
    cex=1,
    log="x",
):
    """
    MA-plot from base means and log fold changes

    A simple helper function that makes a so-called "MA-plot", *i.e.* a scatter
    plot of log2 fold changes (on the y-axis) versus the mean of normalized
    counts (on the x-axis).

    This function also contains the code of the :code:`plotMA` function from
    the :code:`geneplotter` package.

    If :code:`self` contains a column :code:`"svalue"` then these will be used
    for coloring the points (with a default :code:`alpha=0.005`).

    Arguments
    ---------
    alpha : float
        the significance level for thresholding adjusted *p*-values
    main : str, optional
        title for the plot
    xlab : str, optional
        x-axis label, defaults to "mean of normalized counts"
    ylim : pair of floats, optional
        y limits
    colNonSig : str
        color to use for non-significant data points
    colSig : str
        color to use for significant data points
    colLine : str
        color to use for the horizontal (y=0) line
    returnData : bool
        whether to return the DataFrame instead of plotting
    MLE : bool
        if :code:`betaPrior=True` was used, whether to plot the MLE (unshrunken
        estimates), defaults to :code:`False`.  Requires that
        :meth:`.DESeqDataSet.results` was run with :code:`addMLE=True`.  Note
        that the MLE will be plotted regardless of this argument, if
        :func:`DESeq` wasrun with :code:`betaPrior=False`. See
        :func:`lfcShrink` for examples on how to plot shrunken log2 fold
        changes.

    Returns
    -------
    matplotlib.pyplot.Axes
        the axes object to be plotted with :code:`matplotlib.pyplot.show()`
    """
    sval = "svalue" in self.columns

    if sval:
        test_col = "svalue"
    else:
        test_col = "adj_pvalue"

    if MLE:
        if "lfcMLE" not in self.columns:
            raise ValueError(
                "lfcMLE column is not present: you should first run results() with addMLE=True"
            )
        lfc_col = "lfcMLE"
    else:
        lfc_col = "log2FoldChange"

    if alpha is None:
        if sval:
            alpha = 0.005
            log.info("thresholding s-values on alpha=0.005 to color points")
        else:
            if self.alpha is None:
                alpha = 0.1
            else:
                alpha = self.alpha

    isDE = np.where(np.isnan(self[test_col]), False, self[test_col] < alpha)
    df = pd.DataFrame({"mean": self["baseMean"], "lfc": self[lfc_col], "isDE": isDE})

    if returnData:
        return df

    # R code calls geneplotter::plotMA, which is directly ported in Python below
    df = df[df["mean"] != 0]
    py = df["lfc"]
    if ylim is None:
        ylim = np.array([-1, 1]) * np.percentile(np.abs(py[py.is_finite()]), 0.99) * 1.1

    # markers:
    #   - up triangle (6) if below ylim[0]
    #   - down triangle (2) if above ylim[1]
    #   - circle (16) otherwise
    py_lo = py < ylim[0]
    py_hi = py > ylim[1]
    py_in = ~(py_lo | py_hi)

    colors = np.where(df["isDE"], colSig, colNonSig)

    ax = plt.gca()
    ax.scatter(df["mean"][py_in], py[py_in], marker="o", c=colors[py_in], s=cex)
    ax.scatter(
        df["mean"][py_lo],
        np.repeat(ylim[0], py_lo.sum()),
        marker="v",
        c=colors[py_lo],
        s=cex,
    )
    ax.scatter(
        df["mean"][py_hi],
        np.repeat(ylim[1], py_hi.sum()),
        marker="^",
        c=colors[py_hi],
        s=cex,
    )
    ax.axhline(y=0, c=colLine)

    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)

    if "x" in log:
        ax.set_xscale("log")
    if "y" in log:
        ax.set_yscale("log")

    return ax
