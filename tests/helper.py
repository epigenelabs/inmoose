from scipy.stats import nbinom

# mimic R rnbinom
# R (size,prob) parameterization is the same as in scipy.stats
# mu = size (1 - prob) / prob = size (1/prob - 1)
# so mu/size + 1 = 1/prob, hence prob = 1/(1+ mu/size) = size / (size+mu)
def rnbinom(n, size, mu, seed=None):
    p = size / (size + mu)
    return nbinom(size, p).rvs(n, random_state=seed)



