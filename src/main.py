import argparse
import random

import core.linalg as LA
import datasets as D
import functions as F
from utils import load_params, encode_pgm, accuracy


class Model(object):
    def __init__(self, eps0=0.1):
        self.w1, self.b1, self.w2, self.b2, self.w3, self.b3 = load_params()
        self.eps0 = eps0

    def __call__(self, x):
        h = F.relu(x@self.w1.T() + self.b1)
        h = F.relu(h@self.w2.T() + self.b2)
        y = F.softmax(h@self.w3.T() + self.b3)

        return y

    def fgsm(self, x, t):
        d_t = LA.array(
            [[1 if ti == i+1 else 0 for i in range(23)] for ti in t])
        h = -d_t + x
        h = h @ self.w3
        h = h * F.drelu(h)
        h = h @ self.w2
        h = h * F.drelu(h)
        d = h @ self.w1

        return F.sign(d) * self.eps0

    def baseline(self, x):
        eps = LA.array([[random.choice((-1, 1)) for i in range(1024)]
                        for j in range(154)])
        return eps * self.eps0


def main(task):
    if task == 2:
        return task2()
    elif task == 3:
        return task3()
    return task4()


def task2():
    x, t = D.get_pgm()
    M = Model()
    y = M(x)
    y = list(map(lambda yi: F.argmax(yi) + 1, y))

    print(accuracy(y, t))
    return accuracy(y, t)


def task3():
    x, t = D.get_pgm()
    epss = (0.01, 0.05, 0.10, 0.20)
    fgsm = []
    base = []
    for eps0 in epss:
        M = Model(eps0=eps0)
        y = M(x)
        x_f = x + M.fgsm(y, t)
        x_b = x + M.baseline(y)

        if eps0 == 0.1:
            [encode_pgm(x_f[i], i+1) for i in range(x.shape[0])]

        y_f = M(x_f)
        y_f = list(map(lambda yi: F.argmax(yi) + 1, y_f))
        y_b = M(x_b)
        y_b = list(map(lambda yi: F.argmax(yi) + 1, y_b))
        fgsm.append(accuracy(y_f, t))
        base.append(accuracy(y_b, t))
        print(f"eps0:{eps0}, FGSM, accuracy:{fgsm[-1]}")
        print(f"eps0:{eps0}, Base, accuracy:{base[-1]}")

    return fgsm, base


def task4(loop=15):
    x, t = D.get_pgm()
    fgsm = []
    base = []
    M = Model()
    y_f = M(x)
    y_b = M(x)
    for i in range(1, loop+1):
        x_f = x + M.fgsm(y_f, t)
        x_b = x + M.baseline(y_b)

        y_f = M(x_f)
        yidx_f = list(map(lambda yi: F.argmax(yi) + 1, y_f))
        y_b = M(x_b)
        yidx_b = list(map(lambda yi: F.argmax(yi) + 1, y_b))
        fgsm.append(accuracy(yidx_f, t))
        base.append(accuracy(yidx_b, t))
        print(f"times:{i}, FGSM, accuracy:{fgsm[-1]}")
        print(f"times:{i}, Base, accuracy:{base[-1]}")

    [encode_pgm(x_f[i], i+1, dir="problem4") for i in range(x.shape[0])]

    return fgsm, base


def parse_args(parser):
    parser.add_argument('-t', '--task', dest='task', type=int,
                        help='run the Task')
    return parser.parse_args()


if __name__ == "__main__":
    random.seed(0)
    parser = argparse.ArgumentParser(description='Process some integers.')
    args = parse_args(parser)
    main(args.task)
