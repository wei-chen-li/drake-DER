import unittest
from pydrake.common.test_utilities import numpy_compare

from pydrake.autodiffutils import AutoDiffXd
from pydrake.multibody.der import (
    DerStructuralProperty_,
    DerUndeformedState_,
)


class TestMultibodyDer(unittest.TestCase):
    @numpy_compare.check_all_types
    def test_der_structural_property(self, T):
        w, h = 1.0e-3, 2.5e-3
        A = w * h
        I1 = h * w**3 / 12
        I2 = w * h**3 / 12
        J = (h * w**3 + w * h**3) / 12
        E, G = 3e9, 0.8e9
        rho = 910

        prop = DerStructuralProperty_[T].FromRectangularCrossSection(
            width=w, height=h, youngs_modulus=E, shear_modulus=G, mass_density=rho)
        numpy_compare.assert_float_allclose(prop.A(), A)
        numpy_compare.assert_float_allclose(prop.EA(), E * A)
        numpy_compare.assert_float_allclose(prop.EI1(), E * I1)
        numpy_compare.assert_float_allclose(prop.EI2(), E * I2)
        numpy_compare.assert_float_allclose(prop.GJ(), G * J)
        numpy_compare.assert_float_allclose(prop.rhoA(), rho * A)
        numpy_compare.assert_float_allclose(prop.rhoJ(), rho * J)

        A *= 4
        prop.set_A(A)
        numpy_compare.assert_float_allclose(prop.A(), A)

    @numpy_compare.check_all_types
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
