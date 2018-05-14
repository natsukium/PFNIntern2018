import unittest

import core.linalg as LA
from functions.math.sign import sign


XV = LA.array([1, 0.1, 0, -0.1, -1])
XM = LA.array([[1, 0.1, 0], [-0.1, -1, 0]])


class TestSign(unittest.TestCase):
    def test_sign(self):
        self.assertEqual(sign(XV), LA.array([1, 1, 0, -1, -1]))
        self.assertEqual(sign(XM), LA.array([[1, 1, 0], [-1, -1, 0]]))
