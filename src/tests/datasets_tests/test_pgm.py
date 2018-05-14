import unittest

from datasets.pgm import get_pgm


class TestGetPGM(unittest.TestCase):
    def test_get_pgm(self):
        x, t = get_pgm()
        self.assertEqual(x.shape, (154, 1024))
        self.assertEqual(t.shape, 154)
        self.assertTrue(max(x[0]) <= 1)
