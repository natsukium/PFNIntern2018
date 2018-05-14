import unittest

import core.linalg as LA
from utils import load_params, accuracy


Y = LA.array([i for i in range(23)])
T = LA.array([j + j % 2 for j in range(23)])


class TestUtils(unittest.TestCase):
    def test_load_params(self):
        w1, b1, w2, b2, w3, b3 = load_params()
        self.assertEqual(w1.shape, (256, 1024))
        self.assertEqual(b1.shape, (154, 256))
        self.assertEqual(w2.shape, (256, 256))
        self.assertEqual(b2.shape, (154, 256))
        self.assertEqual(w3.shape, (23, 256))
        self.assertEqual(b3.shape, (154, 23))

    def test_accuracy(self):
        self.assertEqual(accuracy(Y, T), 12/23)
