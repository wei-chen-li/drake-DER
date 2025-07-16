#include "drake/multibody/der/elastic_energy.h"

#include <functional>
#include <memory>
#include <optional>
#include <set>
#include <vector>

#include <gtest/gtest.h>

#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/math/autodiff.h"
#include "drake/math/autodiff_gradient.h"

namespace drake {
namespace multibody {
namespace der {
namespace internal {
namespace {

using Eigen::Vector3d;

class ElasticEnergyTest : public ::testing::TestWithParam<bool> {
 protected:
  using T = AutoDiffXd;
  using EnergyCalcFuncType = std::function<T(  //
      const DerStructuralProperty<T>&, const DerUndeformedState<T>&,
      const DerState<T>&)>;
  using JacobianCalcFuncType = std::function<void(
      const DerStructuralProperty<T>&, const DerUndeformedState<T>&,
      const DerState<T>&, EigenPtr<Eigen::VectorX<T>>)>;
  using HessianCalcFuncType = std::function<void(
      const DerStructuralProperty<T>&, const DerUndeformedState<T>&,
      const DerState<T>&, EnergyHessianMatrix<T>*)>;

  void SetUp() override {
    const bool has_closed_ends = GetParam();

    const auto [E, G, rho] = std::make_tuple(3e9, 0.8e9, 910);
    prop_ = DerStructuralProperty<T>::FromRectangularCrossSection(1.38e-3, 6e-3,
                                                                  E, G, rho);
    std::vector<Eigen::Vector3<T>> node_positions;
    std::vector<T> edge_angles;
    const double l = 0.01;
    if (!has_closed_ends) {
      node_positions = {Vector3d(0, 0, 0), Vector3d(l, 0, l * 0.5),
                        Vector3d(l, l, l * 1.5), Vector3d(0, l, l * 3.0)};
      edge_angles = {0, 0.1, 0.2};
      undeformed_ = DerUndeformedState<T>::ZeroCurvatureAndTwist(
          has_closed_ends, Eigen::RowVectorX<T>::Constant(3, l));
    } else {
      node_positions = {Vector3d(0, 0, 0), Vector3d(l, 0, l),
                        Vector3d(l, l, l * 1.5), Vector3d(0, l, l)};
      edge_angles = {0, 0.1, 0.2, 0.1};
      undeformed_ = DerUndeformedState<T>::ZeroCurvatureAndTwist(
          has_closed_ends, Eigen::RowVectorX<T>::Constant(4, l));
    }
    Eigen::Vector3<T> d1_0(0, 1, 0);
    der_state_system_ = std::make_unique<DerStateSystem<T>>(
        has_closed_ends, node_positions, edge_angles, d1_0);

    state_ = std::make_unique<DerState<T>>(der_state_system_.get());

    auto q = math::ExtractValue(state_->get_position());
    state_->AdvancePositionToNextStep(math::InitializeAutoDiff(q));
    state_->FixReferenceFrameDuringAutoDiff();
  }

  void CheckJacobian(EnergyCalcFuncType energy_calc_func,
                     JacobianCalcFuncType jacobian_calc_func) const {
    Eigen::VectorX<T> jacobian(state_->num_dofs());
    jacobian.setZero();
    jacobian_calc_func(*prop_, *undeformed_, *state_, &jacobian);
    auto dEdq = math::ExtractValue(jacobian);

    auto E = energy_calc_func(*prop_, *undeformed_, *state_);
    auto dEdq_autodiff = E.derivatives();

    EXPECT_TRUE(CompareMatrices(dEdq, dEdq_autodiff, 1e-10));
  }

  void CheckHessian(JacobianCalcFuncType jacobian_calc_func,
                    HessianCalcFuncType hessian_calc_func) const {
    EnergyHessianMatrix<T> hessian =
        EnergyHessianMatrix<T>::Allocate(state_->num_dofs());
    hessian_calc_func(*prop_, *undeformed_, *state_, &hessian);
    Eigen::MatrixXd d2Edq2 = math::ExtractValue(hessian.MakeDenseMatrix());

    Eigen::VectorX<T> jacobian(state_->num_dofs());
    jacobian.setZero();
    jacobian_calc_func(*prop_, *undeformed_, *state_, &jacobian);
    auto d2Edq2_autodiff = math::ExtractGradient(jacobian);
    d2Edq2_autodiff =
        (d2Edq2_autodiff + d2Edq2_autodiff.transpose()).eval() / 2;

    EXPECT_TRUE(CompareMatrices(d2Edq2, d2Edq2_autodiff, 1e-8));
  }

  std::optional<DerStructuralProperty<T>> prop_;
  std::optional<DerUndeformedState<T>> undeformed_;
  std::unique_ptr<DerStateSystem<T>> der_state_system_;
  std::unique_ptr<DerState<T>> state_;
};

INSTANTIATE_TEST_SUITE_P(HasClosedEnds, ElasticEnergyTest,
                         ::testing::Values(false, true));

TEST_P(ElasticEnergyTest, StretchingEnergyJacobian) {
  CheckJacobian(&ComputeStretchingEnergy<T>, &AddStretchingEnergyJacobian<T>);
}

TEST_P(ElasticEnergyTest, StretchingEnergyHessian) {
  CheckHessian(&AddStretchingEnergyJacobian<T>, &AddStretchingEnergyHessian<T>);
}

TEST_P(ElasticEnergyTest, TwistingEnergyJacobian) {
  CheckJacobian(&ComputeTwistingEnergy<T>, &AddTwistingEnergyJacobian<T>);
}

TEST_P(ElasticEnergyTest, TwistingEnergyHessian) {
  CheckHessian(&AddTwistingEnergyJacobian<T>, &AddTwistingEnergyHessian<T>);
}

TEST_P(ElasticEnergyTest, BendingEnergyJacobian) {
  CheckJacobian(&ComputeBendingEnergy<T>, &AddBendingEnergyJacobian<T>);
}

TEST_P(ElasticEnergyTest, BendingEnergyHessian) {
  CheckHessian(&AddBendingEnergyJacobian<T>, &AddBendingEnergyHessian<T>);
}

TEST_P(ElasticEnergyTest, TotalElasticEnergyJacobian) {
  CheckJacobian(&ComputeElasticEnergy<T>, &ComputeElasticEnergyJacobian<T>);
}

TEST_P(ElasticEnergyTest, TotalElasticEnergyHessian) {
  CheckHessian(&ComputeElasticEnergyJacobian<T>,
               &ComputeElasticEnergyHessian<T>);
}

}  // namespace
}  // namespace internal
}  // namespace der
}  // namespace multibody
}  // namespace drake
