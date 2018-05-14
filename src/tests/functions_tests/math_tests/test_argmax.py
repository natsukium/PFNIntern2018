import unittest

import core.linalg as LA
from functions.math.argmax import argmax


X = LA.array([0.1, 0.4, 0.8, 0.3, 0.5])


class TestArgmax(unittest.TestCase):
    def test_argmax(self):
        self.assertEqual(argmax(X), 2)
