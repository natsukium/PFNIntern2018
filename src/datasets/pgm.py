from pathlib import Path

import core.linalg as LA


def get_pgm():
    p = [Path(f"../ml/pgm/{i}.pgm") for i in range(1, 155)]

    batch = []
    for img in p:
        with img.open() as f:
            x = f.readlines()
        vector = []
        for xi in x[3:]:
            vector += list(map(int, xi.split()))
        batch.append(vector)

    X = LA.array(batch) / 255

    q = Path("../ml/labels.txt")
    with q.open() as f:
        t = f.readlines()
    T = LA.array(list(map(int, t)))

    return X, T
