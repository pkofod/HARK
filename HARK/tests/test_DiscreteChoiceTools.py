"""
This file implements unit tests to check HARK/utilities.py
"""
from __future__ import print_function, division
from __future__ import absolute_import

from builtins import str
from builtins import zip
from builtins import range
from builtins import object

# Bring in modules we need
import unittest
import numpy as np
from HARK.interpolation import discreteEnvelope

class testsForDiscreteChoiceTools(unittest.TestCase):

    #def setUp(self):
    #    return

    def test_NoTasteShocks(self):

        Vs = np.array([[[0, 1], [9, 0]], [[1,2], [1, 3]]])
        sigma = 0.0

        Vref = np.array([[1, 2], [9, 3]])
        Pref = np.array([[[0, 0], [1, 0]], [[1, 1], [0, 1]]])
        V, P = discreteEnvelope(Vs, sigma)

        self.assertEqual((Vref == V).all(), True)
        self.assertEqual((Pref == P).all(), True)
