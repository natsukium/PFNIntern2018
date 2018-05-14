from math import log, exp


def cross_entropy(x, t):
    y = [[exp(xij) for xij in xi] for xi in x]
    return [-xi[ti-1] + log(sum(yi)) for xi, yi, ti in zip(x, y, t)]
