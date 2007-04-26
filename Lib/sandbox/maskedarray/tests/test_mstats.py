# pylint: disable-msg=W0611, W0612, W0511,R0201
"""Tests suite for maskedArray statistics.

:author: Pierre Gerard-Marchant
:contact: pierregm_at_uga_dot_edu
:version: $Id$
"""
__author__ = "Pierre GF Gerard-Marchant ($Author$)"
__version__ = '1.0'
__revision__ = "$Revision$"
__date__     = '$Date$'

import numpy

import maskedarray
from maskedarray import masked, masked_array

import maskedarray.testutils
from maskedarray.testutils import *

from maskedarray.mstats import mquantiles, mmedian

#..............................................................................
class test_quantiles(NumpyTestCase):
    "Base test class for MaskedArrays."
    def __init__(self, *args, **kwds):
        NumpyTestCase.__init__(self, *args, **kwds)
        self.a = maskedarray.arange(1,101)
    #
    def test_1d_nomask(self):
        "Test quantiles 1D - w/o mask."
        a = self.a
        assert_almost_equal(mquantiles(a, alphap=1., betap=1.), 
                            [25.75, 50.5, 75.25])
        assert_almost_equal(mquantiles(a, alphap=0, betap=1.), 
                            [25., 50., 75.])
        assert_almost_equal(mquantiles(a, alphap=0.5, betap=0.5), 
                            [25.5, 50.5, 75.5])
        assert_almost_equal(mquantiles(a, alphap=0., betap=0.), 
                            [25.25, 50.5, 75.75])
        assert_almost_equal(mquantiles(a, alphap=1./3, betap=1./3), 
                            [25.41666667, 50.5, 75.5833333])
        assert_almost_equal(mquantiles(a, alphap=3./8, betap=3./8), 
                            [25.4375, 50.5, 75.5625])
        assert_almost_equal(mquantiles(a), [25.45, 50.5, 75.55])# 
    #
    def test_1d_mask(self):
        "Test quantiles 1D - w/ mask."
        a = self.a
        a[1::2] = masked
        assert_almost_equal(mquantiles(a, alphap=1., betap=1.), 
                            [25.5, 50.0, 74.5])
        assert_almost_equal(mquantiles(a, alphap=0, betap=1.), 
                            [24., 49., 74.])
        assert_almost_equal(mquantiles(a, alphap=0.5, betap=0.5), 
                            [25., 50., 75.])
        assert_almost_equal(mquantiles(a, alphap=0., betap=0.), 
                            [24.5, 50.0, 75.5])
        assert_almost_equal(mquantiles(a, alphap=1./3, betap=1./3), 
                            [24.833333, 50.0, 75.166666])
        assert_almost_equal(mquantiles(a, alphap=3./8, betap=3./8), 
                            [24.875, 50., 75.125])
        assert_almost_equal(mquantiles(a), [24.9, 50., 75.1])
    #
    def test_2d_nomask(self):
        "Test quantiles 2D - w/o mask."
        a = self.a
        b = maskedarray.resize(a, (100,100))
        assert_almost_equal(mquantiles(b), [25.45, 50.5, 75.55])
        assert_almost_equal(mquantiles(b, axis=0), maskedarray.resize(a,(3,100)))
        assert_almost_equal(mquantiles(b, axis=1), 
                            maskedarray.resize([25.45, 50.5, 75.55], (100,3)))
    #
    def test_2d_mask(self):
        "Test quantiles 2D - w/ mask."
        a = self.a
        a[1::2] = masked
        b = maskedarray.resize(a, (100,100))
        assert_almost_equal(mquantiles(b), [25., 50., 75.])
        assert_almost_equal(mquantiles(b, axis=0), maskedarray.resize(a,(3,100)))
        assert_almost_equal(mquantiles(b, axis=1), 
                            maskedarray.resize([24.9, 50., 75.1], (100,3)))        
        
class test_median(NumpyTestCase):
    def __init__(self, *args, **kwds):
        NumpyTestCase.__init__(self, *args, **kwds)
        
    def test_2d(self):
        "Tests median w/ 2D"
        (n,p) = (101,30)
        x = masked_array(numpy.linspace(-1.,1.,n),)
        x[:10] = x[-10:] = masked
        z = masked_array(numpy.empty((n,p), dtype=numpy.float_))
        z[:,0] = x[:]
        idx = numpy.arange(len(x))
        for i in range(1,p):
            numpy.random.shuffle(idx)
            z[:,i] = x[idx]
        assert_equal(mmedian(z[:,0]), 0)
        assert_equal(mmedian(z), numpy.zeros((p,)))
        
    def test_3d(self):
        "Tests median w/ 3D"
        x = maskedarray.arange(24).reshape(3,4,2)
        x[x%3==0] = masked
        assert_equal(mmedian(x), [[12,9],[6,15],[12,9],[18,15]])
        x.shape = (4,3,2)
        assert_equal(mmedian(x),[[99,10],[11,99],[13,14]])
        x = maskedarray.arange(24).reshape(4,3,2)
        x[x%5==0] = masked
        assert_equal(mmedian(x), [[12,10],[8,9],[16,17]])
        
###############################################################################
#------------------------------------------------------------------------------
if __name__ == "__main__":
    NumpyTest().run()
            