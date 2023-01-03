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
