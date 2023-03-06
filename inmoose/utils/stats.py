from scipy.stats import nbinom

def rnbinom(n, size, mu, seed=None):
    """mimic R rnbinom function, to draw samples from a Negative Binomial distribution.

    The (size, prob) parameterization used in R is the same as in scipy.stats:
        mu = size * (1 - prob) / prob = size * (1/prob - 1)
    so
        mu / size + 1 = 1/prob
    hence
        prob = 1 / (1 + mu/size) = size / (size + mu)

    Arguments
    ---------
    n : int or tuple of ints
        shape of the output. If n = (n1, n2, ..., np) then n1*n2*...*np random samples are drawn.
    size : float or array-like
        size parameter of the Negative Binomial distribution.
        all values must be positive
    mu : float or array-like
        mean parameter of the Negative Binomial distribution
        all values must be positive
    seed : int, optional
        pass a seed to the underlying RNG. If `None`, then the RNG is seeded using unpredictable entropy from the system.
        See the documentation of scipy.stats about RNG seeding for more details.
    """
    p = size / (size + mu)
    return nbinom(size, p).rvs(n, random_state=seed)
