import unittest

import core.linalg as LA
from functions.activation.softmax import softmax


XV = LA.array([0.5, 0.7, 0.3])
XM = LA.array([[0.5, 0.7, 0.3], [0.2, 0.6, 0.8]])
# chainer.functions.softmaxより
SV = LA.array([0.32893292, 0.40175958, 0.2693075])
SM = LA.array([[0.32893292, 0.40175958, 0.2693075],
               [0.23180647, 0.34581461, 0.42237892]])
EPS = 1e-5


class TestSoftmax(unittest.TestCase):
    def test_softmax(self):
        self.assertTrue(max(softmax(XV) - SV) < EPS)
        self.assertTrue(max(max(softmax(XM) - SM)) < EPS)
