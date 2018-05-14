from pathlib import Path

import core.linalg as LA


def load_params():
    p = Path("../ml/param.txt")
    with p.open() as f:
        params = f.readlines()

    h = 256
    c = 23
    w1 = LA.array([list(map(float, param.split())) for param in params[:h]])
    b1 = LA.array([list(map(float, params[h].split()))] * 154)
    w2 = LA.array(
        [list(map(float, param.split())) for param in params[h+1:h*2+1]])
    b2 = LA.array([list(map(float, params[h*2+1].split()))] * 154)
    w3 = LA.array(
        [list(map(float, param.split())) for param in params[h*2+2:(h+1)*2+c]])
    b3 = LA.array([list(map(float, params[(h+1)*2+c].split()))] * 154)

    return w1, b1, w2, b2, w3, b3


def encode_pgm(x, n, dir="problem3"):
    # normalization
    y = (x - min(x))/(max(x) - min(x))
    y = list(map(int, y * 255))

    s = "P2\n32 32\n255\n"
    for i, st in enumerate(y):
        if i % 32 == 31:
            s += f"{st}\n"
        else:
            s += f"{st} "

    p = Path(f"../{dir}/{n}.pgm")
    if not p.exists():
        with p.open("x") as f:
            f.write(s)


def accuracy(y, t):
    accuracy = 0
    for yi, ti in zip(y, t):
        if yi == ti:
            accuracy += 1

    return accuracy / len(t)
