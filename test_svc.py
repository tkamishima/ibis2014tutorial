#!/usr/bin/env python
# -*- coding: utf-8 -*-

from numpy.testing import (
    assert_array_equal,
    assert_array_less,
    assert_allclose,
    assert_array_max_ulp,
    assert_array_almost_equal_nulp)
import unittest
import numpy as np

##### Test Classes #####

class Test_SVC(unittest.TestCase):

    def test_fit(self):
        from sklearn import svm
        from sklearn import datasets
        iris = datasets.load_iris()
        clf = svm.SVC(kernel='linear', C=1)
        clf.fit(iris.data[50:150, :], iris.target[50:150])

        np.testing.assert_allclose(clf.coef_,
            [[-0.59549776, -0.9739003 ,  2.03099958,  2.00630267]],
            rtol=1e-5)
        classes = clf.predict(iris.data[[50, 51, 100, 101], :])
        np.testing.assert_array_equal(classes, [1, 1, 2, 2])

##### Main routine #####
if __name__ == '__main__':
    unittest.main()
