# -----------------------------------------------------------------------------
# Copyright (C) 2004-2022 Gordon Smyth, Yifang Hu, Matthew Ritchie, Jeremy Silver, James Wettenhall, Davis McCarthy, Di Wu, Wei Shi, Belinda Phipson, Aaron Lun, Natalie Thorne, Alicia Oshlack, Carolyn de Graaf, Yunshun Chen, Mette Langaas, Egil Ferkingstad, Marcus Davy, Francois Pepin, Dongseok Choi
# Copyright (C) 2024 Maximilien Colange

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

# This file is based on the file 'R/dups.R' of the Bioconductor limma package (version 3.55.1).

import numpy as np


def unwrapdups(M, ndups=2, spacing=1):
    """
    Unwraps a matrix (M) for a series of experiments, combining duplicate spots for each gene into a single row.

    Arguments
    ---------
    M : ndarray
        the input matrix containing data for multiple experiments.
    ndups : int, optional
        the number of duplicate spots for each gene. Defaults to 2.
    spacing : int, optional
        the number of rows between duplicate measurements for a gene. Defaults to 1.

    Returns
    -------
    ndarray
        the unwrapped matrix with duplicate spots combined.
    """
    if ndups <= 0:
        raise ValueError("ndups must be strictly positive")
    if ndups == 1:
        return M

    M = np.asarray(M)  # Convert to NumPy array
    if M.ndim == 1:
        M = M[:, None]
    nspots, nslides = M.shape  # Get number of spots and slides

    ngroups = nspots // (ndups * spacing)
    new_shape = (spacing, ndups, ngroups, nslides)
    M = M.reshape(new_shape, order="F")

    # Transpose to combine duplicate spots for each gene
    M = np.transpose(M, (0, 2, 1, 3))
    return M.reshape(
        spacing * ngroups, ndups * nslides, order="F"
    )  # Reshape final output


def uniquegenelist(genelist, ndups=2, spacing=1):
    """
    Eliminates entries in a gene list for duplicate spots based on unwrapped data.

    Arguments
    ---------
    genelist : array_like
        The list of genes with potentially duplicate spots.
    ndups : int, optional
        The number of duplicate spots for each gene. Defaults to 2.
    spacing : int, optional
        The number of rows between duplicate measurements for a gene. Defaults to 1.

    Returns
    -------
    list
        The list of genes with duplicate spots removed.
    """
    if ndups <= 1:
        return genelist

    i = np.squeeze(
        unwrapdups(np.arange(genelist.shape[0]), ndups=ndups, spacing=spacing)[:, 0]
    )
    if i.ndim == 0:
        i = i.reshape(1)
    if genelist.ndim == 1:
        return genelist[i]
    else:
        return genelist[i, :]
