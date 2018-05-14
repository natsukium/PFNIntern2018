import unittest
from math import exp, log

import core.linalg as LA
from functions.loss.cross_entropy import cross_entropy


X = LA.array([[0.5, 0.3, 0.6]])
T = LA.array([2])


class TestCrossEntropy(unittest.TestCase):
    def test_cross_entropy(self):
        sum_exp = exp(X[0, 0]) + exp(X[0, 1]) + exp(X[0, 2])
        entropy = -X[0, T[0]-1] + log(sum_exp)
        self.assertEqual(cross_entropy(X, T)[0], entropy)
