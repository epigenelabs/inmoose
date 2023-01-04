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

# This file is based on the file 'R/q2qnbinom.R' of the Bioconductor edgeR package (version 3.38.4).


import numpy as np

from .stats import pnorm, qnorm, pgamma, qgamma

def q2qpois(z, input_mean, output_mean):
    """
    Approximate quantile to quantile mapping between Poisson distributions
    """
    raise RuntimeError("unimplemented")

def q2qnbinom(x, input_mean, output_mean, dispersion=0):
    """
    Approximate quantile to quantile mapping between negative-binomial distributions with different means but same dispersion
    """
    if (x < 0).any():
        raise ValueError("x must be non-negative")
    if (input_mean < 0).any():
        raise ValueError("input_mean must be non-negative")
    if (output_mean < 0).any():
        raise ValueError("output_mean must be non-negative")
    if (np.asanyarray(dispersion) < 0).any():
        raise ValueError("dispersion must be non-negative")

    eps = 1e-14
    zero = np.logical_or(input_mean < eps, output_mean < eps)
    input_mean[zero] += 0.25
    output_mean[zero] += 0.25
    ri = 1 + dispersion*input_mean
    vi = input_mean*ri
    ro = 1 + dispersion*output_mean
    vo = output_mean*ro
    i = (x >= input_mean)
    j = np.logical_not(i)
    p1 = x.copy()
    p2 = x.copy()
    q1 = x.copy()
    q2 = x.copy()
    if i.any():
        p1[i] = pnorm(x[i], mean=input_mean[i], sd=np.sqrt(vi[i]), lower_tail=False, log_p=True)
        p2[i] = pgamma(x[i], shape=input_mean[i]/ri[i], scale=ri[i], lower_tail=False, log_p=True)
        q1[i] = qnorm(p1[i], mean=output_mean[i], sd=np.sqrt(vo[i]), lower_tail=False, log_p=True)
        q2[i] = qgamma(p2[i], shape=output_mean[i]/ro[i], scale=ro[i], lower_tail=False, log_p=True)

    if j.any():
        p1[j] = pnorm(x[j], mean=input_mean[j], sd=np.sqrt(vi[j]), lower_tail=True, log_p=True)
        p2[j] = pgamma(x[j], shape=input_mean[j]/ri[j], scale=ri[j], lower_tail=True, log_p=True)
        q1[j] = qnorm(p1[j], mean=output_mean[j], sd=np.sqrt(vo[j]), lower_tail=True, log_p=True)
        q2[j] = qgamma(p2[j], shape=output_mean[j]/ro[j], scale=ro[j], lower_tail=True, log_p=True)

    return (q1+q2)/2
