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


def plotDispEsts_dds(
    self,
    ymin=None,
    CV=False,
    genecol="black",
    fitcol="red",
    finalcol="dodgerblue",
    legend=True,
    xlab=None,
    ylab=None,
    log="xy",
    cex=0.7,
):
    """
    A simple helper function that plots the per-gene dispersion estimates
    together with the fitted mean-dispersion relationship.

    Arguments
    ---------
    ymin
        the lower bound for points on the plot, points beyond this are drawn as
        triangles at :code:`ymin`
    CV : bool
        whether to plot the asymptotic or biological coefficient of variation
        (the square root of dispersion) on the y-axis. As the mean grows to
        infinity, the square root of dispersion gives the coefficient of
        variation for the counts. Default is :code:`False`, plotting
        dispersion.
    genecol : str
        the color for gene-wise dispersion estimates
    fitcol : str
        the color of the fitted estimates
    finalcol : str
        the color of the final estimates used for testing
    legend : bool
        whether to draw a legend
    xlab : str
    ylab : str
    log : {"", "x", "y", "xy"}
        the axis (if any) to log scale
    cex : float
        the default size of the points to plot
    """

    if xlab is None:
        xlab = "mean of normalized counts"
    if ylab is None:
        if CV:
            ylab = "coefficient of variation"
        else:
            ylab = "dispersion"

    px = self.var["baseMean"]
    sel = px > 0
    px = px[sel]

    # transformation of dispersion into CV or not
    if CV:
        f = np.sqrt
    else:

        def f(x):
            return x

    py = f(self.var["dispGeneEst"][sel])
    if ymin is None:
        ymin = 10 ** np.floor(np.log10(np.nanmin(py[py > 0])) - 0.1)

    plt.scatter(px, np.maximum(py, ymin), s=cex, c=genecol, label="gene-est")
    if "x" in log:
        plt.xscale("log")
    if "y" in log:
        plt.yscale("log")

    outliers = (self.var["dispOutlier"] == 1.0)[sel]

    if self.dispersions is not None:
        plt.scatter(
            px[~outliers],
            f(self.dispersions[sel][~outliers]),
            c=finalcol,
            s=cex,
            label="final",
        )
        # use a circle over outliers
        plt.scatter(
            px[outliers],
            f(self.dispersions[sel][outliers]),
            s=8 * cex,
            marker="o",
            facecolors="none",
            edgecolors=finalcol,
        )

    if self.var["dispFit"] is not None:
        plt.scatter(px, f(self.var["dispFit"][sel]), c=fitcol, s=cex, label="fitted")

    if legend:
        plt.legend(loc="lower right")

    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.show()


def plotMA_dds(
    self,
    alpha=0.1,
    main="",
    xlab="mean of normalized counts",
    ylim=None,
    MLE=False,
    *args,
    **kwargs,
):
    """
    Helper function to MA-plot the results of :code:`self`
    """
    res = self.results(*args, **kwargs)
    return res.plotMA(alpha=alpha, main=main, xlab=xlab, ylim=ylim, MLE=MLE)
