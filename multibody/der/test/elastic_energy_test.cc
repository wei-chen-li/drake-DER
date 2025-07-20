#include "drake/multibody/der/elastic_energy.h"

#include <functional>
#include <memory>
#include <optional>
#include <set>
#include <vector>

#include <gtest/gtest.h>

#include "drake/common/pointer_cast.h"
#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/common/test_utilities/limit_malloc.h"
#include "drake/math/autodiff.h"
#include "drake/math/autodiff_gradient.h"

namespace drake {
namespace multibody {
namespace der {
namespace internal {
namespace {

using Eigen::Vector3d;
using test::LimitMalloc;

class ElasticEnergyTest : public ::testing::TestWithParam<bool> {
 protected:
  template <typename T>
  using EnergyCalcFuncType = std::function<T(  //
      const DerStructuralProperty<T>&, const DerUndeformedState<T>&,
      const DerState<T>&)>;

  template <typename T>
  using JacobianCalcFuncType = std::function<void(
      const DerStructuralProperty<T>&, const DerUndeformedState<T>&,
      const DerState<T>&, EigenPtr<Eigen::VectorX<T>>)>;

  template <typename T>
  using HessianCalcFuncType = std::function<void(
      const DerStructuralProperty<T>&, const DerUndeformedState<T>&,
      const DerState<T>&, EnergyHessianMatrix<T>*, Parallelism)>;

  void SetUp() override {
    const bool has_closed_ends = GetParam();

    const auto [E, G, rho] = std::make_tuple(3e9, 0.8e9, 910);
    prop_ad_ = DerStructuralProperty<AutoDiffXd>::FromRectangularCrossSection(
        1.38e-3, 6e-3, E, G, rho);

    std::vector<Eigen::Vector3<AutoDiffXd>> node_positions;
    std::vector<AutoDiffXd> edge_angles;
    const double l = 0.01;
    if (!has_closed_ends) {
      node_positions = {Vector3d(0, 0, 0), Vector3d(l, 0, l * 0.5),
                        Vector3d(l, l, l * 1.5), Vector3d(0, l, l * 3.0)};
      edge_angles = {0, 0.1, 0.2};
      undeformed_ad_ = DerUndeformedState<AutoDiffXd>::ZeroCurvatureAndTwist(
          has_closed_ends, Eigen::RowVectorX<AutoDiffXd>::Constant(3, l));
    } else {
      node_positions = {Vector3d(0, 0, 0), Vector3d(l, 0, l),
                        Vector3d(l, l, l * 1.5), Vector3d(0, l, l)};
      edge_angles = {0, 0.1, 0.2, 0.1};
      undeformed_ad_ = DerUndeformedState<AutoDiffXd>::ZeroCurvatureAndTwist(
          has_closed_ends, Eigen::RowVectorX<AutoDiffXd>::Constant(4, l));
    }
    Eigen::Vector3<AutoDiffXd> d1_0(0, 1, 0);
    der_state_system_ad_ = std::make_unique<DerStateSystem<AutoDiffXd>>(
        has_closed_ends, node_positions, edge_angles, d1_0);

    state_ad_ =
        std::make_unique<DerState<AutoDiffXd>>(der_state_system_ad_.get());

    auto q = math::ExtractValue(state_ad_->get_position());
    state_ad_->AdvancePositionToNextStep(math::InitializeAutoDiff(q));
    state_ad_->FixReferenceFrameDuringAutoDiff();

    prop_ = prop_ad_->ToScalarType<double>();
    undeformed_ = undeformed_ad_->ToScalarType<double>();
    der_state_system_ = dynamic_pointer_cast<DerStateSystem<double>>(
        der_state_system_ad_->ToScalarType<double>());
    state_ = std::make_unique<DerState<double>>(der_state_system_.get());
  }

  void CheckJacobian(EnergyCalcFuncType<AutoDiffXd> energy_calc_func,
                     JacobianCalcFuncType<double> jacobian_calc_func) const {
    Eigen::VectorXd dEdq = Eigen::VectorXd::Zero(state_ad_->num_dofs());
    {
      LimitMalloc guard;
      jacobian_calc_func(*prop_, *undeformed_, *state_, &dEdq);
    }

    auto E = energy_calc_func(*prop_ad_, *undeformed_ad_, *state_ad_);
    auto dEdq_autodiff = E.derivatives();

    EXPECT_TRUE(CompareMatrices(dEdq, dEdq_autodiff, 1e-10));
  }

  void CheckHessian(JacobianCalcFuncType<AutoDiffXd> jacobian_calc_func,
                    HessianCalcFuncType<double> hessian_calc_func) const {
    EnergyHessianMatrix<double> d2Edq2 =
        EnergyHessianMatrix<double>::Allocate(state_ad_->num_dofs());
    {
      /* OpenMP runtime allocates heap, so we disarm LimitMalloc if _OPENMP is
       defined. */
#if !defined(_OPENMP)
      LimitMalloc guard;
#endif
      hessian_calc_func(*prop_, *undeformed_, *state_, &d2Edq2,
                        Parallelism::None());
    }

    Eigen::VectorX<AutoDiffXd> jacobian =
        Eigen::VectorX<AutoDiffXd>::Zero(state_ad_->num_dofs());
    jacobian_calc_func(*prop_ad_, *undeformed_ad_, *state_ad_, &jacobian);
    auto d2Edq2_autodiff = math::ExtractGradient(jacobian);
    d2Edq2_autodiff =
        (d2Edq2_autodiff + d2Edq2_autodiff.transpose()).eval() / 2;

    EXPECT_TRUE(
        CompareMatrices(d2Edq2.MakeDenseMatrix(), d2Edq2_autodiff, 1e-8));

    /* Also test parallelized version. */
    d2Edq2.SetZero();
    {
#if !defined(_OPENMP)
      LimitMalloc guard;
#endif
      hessian_calc_func(*prop_, *undeformed_, *state_, &d2Edq2,
                        Parallelism::Max());
    }
    EXPECT_TRUE(
        CompareMatrices(d2Edq2.MakeDenseMatrix(), d2Edq2_autodiff, 1e-8));
  }

  std::optional<DerStructuralProperty<AutoDiffXd>> prop_ad_;
  std::optional<DerUndeformedState<AutoDiffXd>> undeformed_ad_;
  std::unique_ptr<DerStateSystem<AutoDiffXd>> der_state_system_ad_;
  std::unique_ptr<DerState<AutoDiffXd>> state_ad_;

  std::optional<DerStructuralProperty<double>> prop_;
  std::optional<DerUndeformedState<double>> undeformed_;
  std::unique_ptr<DerStateSystem<double>> der_state_system_;
  std::unique_ptr<DerState<double>> state_;
};

INSTANTIATE_TEST_SUITE_P(HasClosedEnds, ElasticEnergyTest,
                         ::testing::Values(false, true));

TEST_P(ElasticEnergyTest, StretchingEnergyJacobian) {
  CheckJacobian(&ComputeStretchingEnergy<AutoDiffXd>,
                &AddStretchingEnergyJacobian<double>);
}

TEST_P(ElasticEnergyTest, StretchingEnergyHessian) {
  CheckHessian(&AddStretchingEnergyJacobian<AutoDiffXd>,
               &AddStretchingEnergyHessian<double>);
}

TEST_P(ElasticEnergyTest, TwistingEnergyJacobian) {
  CheckJacobian(&ComputeTwistingEnergy<AutoDiffXd>,
                &AddTwistingEnergyJacobian<double>);
}

TEST_P(ElasticEnergyTest, TwistingEnergyHessian) {
  CheckHessian(&AddTwistingEnergyJacobian<AutoDiffXd>,
               &AddTwistingEnergyHessian<double>);
}

TEST_P(ElasticEnergyTest, BendingEnergyJacobian) {
  CheckJacobian(&ComputeBendingEnergy<AutoDiffXd>,
                &AddBendingEnergyJacobian<double>);
}

TEST_P(ElasticEnergyTest, BendingEnergyHessian) {
  CheckHessian(&AddBendingEnergyJacobian<AutoDiffXd>,
               &AddBendingEnergyHessian<double>);
}

TEST_P(ElasticEnergyTest, TotalElasticEnergyJacobian) {
  CheckJacobian(&ComputeElasticEnergy<AutoDiffXd>,
                &ComputeElasticEnergyJacobian<double>);
}

TEST_P(ElasticEnergyTest, TotalElasticEnergyHessian) {
  CheckHessian(&ComputeElasticEnergyJacobian<AutoDiffXd>,
               &ComputeElasticEnergyHessian<double>);
}

}  // namespace
}  // namespace internal
}  // namespace der
}  // namespace multibody
}  // namespace drake
