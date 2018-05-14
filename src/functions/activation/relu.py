import core.linalg as LA


def relu(x):
    if x._is_vector():
        return LA.array([xi * (xi > 0) for xi in x])
    return LA.array([[xij * (xij > 0) for xij in xi] for xi in x])


def drelu(x):
    if x._is_vector():
        return LA.array([xi > 0 for xi in x])
    return LA.array([[xij > 0 for xij in xi] for xi in x])
