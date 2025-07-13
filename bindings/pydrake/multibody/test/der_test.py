import unittest
from pydrake.common.test_utilities import numpy_compare

from pydrake.autodiffutils import AutoDiffXd
from pydrake.multibody.der import (
    DerUndeformedState_
)


class TestMultibodyDer(unittest.TestCase):
    @numpy_compare.check_nonsymbolic_types
    def test_der_undeformed_state(self, T):
        undeformed = DerUndeformedState_[T].ZeroCurvatureAndTwist(
            has_closed_ends=False, edge_length=[[0.10, 0.12, 0.08]])
        numpy_compare.assert_float_equal(
            undeformed.get_edge_length(), [0.10, 0.12, 0.08])
        numpy_compare.assert_float_equal(
            undeformed.get_voronoi_length(), [0.11, 0.10])
        numpy_compare.assert_float_equal(
            undeformed.get_curvature_kappa1(), [0.0, 0.0])
        numpy_compare.assert_float_equal(
            undeformed.get_curvature_kappa2(), [0.0, 0.0])
        numpy_compare.assert_float_equal(
            undeformed.get_twist(), [0.0, 0.0])
