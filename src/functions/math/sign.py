import core.linalg as LA


def sign(x):
    if x._is_vector():
        return LA.array([1 if xi > 0 else -1 if xi < 0 else 0 for xi in x])
    return LA.array(
        [[1 if xij > 0 else -1 if xij < 0 else 0 for xij in xi] for xi in x])
