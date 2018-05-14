import unittest

import core.linalg as LA
from functions.activation.relu import relu, drelu


XV = LA.array([2, 0, -1])
XM = LA.array([[1, 0, -1], [0.1, 0.0, -0.1]])


class TestReLU(unittest.TestCase):
    def test_relu(self):
        self.assertEqual(relu(XV), LA.array([2, 0, 0]))
        self.assertEqual(relu(XM), LA.array([[1, 0, 0], [0.1, 0, 0]]))

    def test_drelu(self):
        self.assertEqual(drelu(XV), LA.array([1, 0, 0]))
        self.assertEqual(drelu(XM), LA.array([[1, 0, 0], [1, 0, 0]]))
