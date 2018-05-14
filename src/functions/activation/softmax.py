from math import exp

import core.linalg as LA


def softmax(x):
    if x._is_vector():
        y = LA.array([exp(xi) for xi in x])
        return y / sum(y)
    y = [[exp(xij) for xij in xi] for xi in x]
    return LA.array([list(LA.array(yi) / sum(yi)) for yi in y])
