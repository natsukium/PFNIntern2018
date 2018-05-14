import unittest

import core.linalg as LA


XV = LA.array([5, 0, -5])
YV = LA.array([1., 1., 1.])
XM = LA.array([[10, 20, 30], [40, 0, 50]])
YM = LA.array([[1., 1., 1.], [1., 1., 1.]])


class TestArray(unittest.TestCase):
    def test_add(self):
        with self.assertRaises(ValueError):
            XV + YM
        self.assertEqual(XV+YV, LA.array([6., 1., -4.]))
        self.assertEqual(XM+YM, LA.array([[11., 21, 31.], [41., 1., 51]]))

    def test_sub(self):
        self.assertEqual(XV-5, LA.array([0, -5, -10]))
        self.assertEqual(XM-YM, LA.array([[9, 19, 29], [39, -1, 49]]))

    def test_multiply(self):
        self.assertEqual(XV*3, LA.array([15, 0, -15]))
        self.assertEqual(XM*3, LA.array([[30, 60, 90], [120, 0, 150]]))
        self.assertEqual(XM*YM, LA.array([[10, 20, 30], [40, 0, 50]]))

    def test_dot(self):
        with self.assertRaises(ValueError):
            XM @ YM
        self.assertEqual(XV@YV, 0)
        self.assertEqual(XM@XV, LA.array([-100, -50]))
        self.assertEqual(
            XM.T()@YM, LA.array([[50, 50, 50], [20, 20, 20], [80, 80, 80]]))

    def test_div(self):
        self.assertEqual(XV/2, LA.array([2.5, 0., -2.5]))
        self.assertEqual(XM/2, LA.array([[5, 10, 15], [20, 0, 25]]))

    def test_transpose(self):
        self.assertEqual(
            XM.transpose(), LA.array([[10, 40], [20, 0], [30, 50]]))
