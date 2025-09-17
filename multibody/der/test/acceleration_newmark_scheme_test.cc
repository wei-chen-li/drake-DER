#include "drake/multibody/der/acceleration_newmark_scheme.h"

#include <limits>
#include <memory>
#include <vector>

#include <gtest/gtest.h>

#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/common/test_utilities/limit_malloc.h"
#include "drake/multibody/der/acceleration_newmark_scheme.h"
#include "drake/multibody/der/velocity_newmark_scheme.h"

namespace drake {
namespace multibody {
namespace der {
namespace internal {
namespace {

using Eigen::Vector3d;
using Eigen::VectorXd;
using test::LimitMalloc;

class AccelerationNewmarkSchemeTest : public ::testing::TestWithParam<double> {
 protected:
  void SetUp() {
    bool has_closed_ends = false;
    std::vector<Vector3d> node_positions = {
        Vector3d(0, 0, 0), Vector3d(1, 0, 0), Vector3d(2, 1, 0)};
    std::vector<double> edge_angles = {M_PI / 6, M_PI / 6};
    der_state_system_ = std::make_unique<DerStateSystem<double>>(
        has_closed_ends, node_positions, edge_angles, std::nullopt);

    scheme_.set_dt(kDt);
  }

  const double kDt = GetParam();
  const double kGamma = 0.6;
  const double kBeta = 0.3;
  const double kTolerance = 16.0 * std::numeric_limits<double>::epsilon();

  AccelerationNewmarkScheme<double> scheme_{kDt, kGamma, kBeta};
  std::unique_ptr<DerStateSystem<double>> der_state_system_;
};

INSTANTIATE_TEST_SUITE_P(TimeStep, AccelerationNewmarkSchemeTest,
                         ::testing::Values(1e-3, 1e-4));

/* Verify that the weights returned by the accessor match expectation. */
TEST_P(AccelerationNewmarkSchemeTest, Weights) {
  EXPECT_EQ(scheme_.GetWeights(),
            (std::array<double, 3>{kBeta * kDt * kDt, kGamma * kDt, 1.0}));
}

/* Verify that the unknowns are the accelerations. */
TEST_P(AccelerationNewmarkSchemeTest, Unknowns) {
  const DerState<double> state(der_state_system_.get());
  EXPECT_EQ(&scheme_.GetUnknowns(state), &state.get_acceleration());
}

/* Tests that AccelerationNewmarkScheme reproduces analytical solutions with
 constant acceleration. */
TEST_P(AccelerationNewmarkSchemeTest, AdvanceDt) {
  DerState<double> state_n(der_state_system_.get());
  const VectorXd a = VectorXd::LinSpaced(state_n.num_dofs(), 0.0, 1.0);
  state_n.SetAcceleration(a);

  const int kTimeSteps = 10;
  for (int i = 0; i < kTimeSteps; ++i) {
    DerState<double> state_np1(der_state_system_.get());
    LimitMalloc guard;
    scheme_.AdvanceDt(state_n, a, &state_np1);
    state_n.CopyFrom(state_np1);
  }

  const double total_time = kDt * kTimeSteps;
  const DerState<double> state_0(der_state_system_.get());
  EXPECT_TRUE(CompareMatrices(state_n.get_acceleration(), a));
  EXPECT_TRUE(CompareMatrices(state_n.get_velocity(),
                              state_0.get_velocity() + total_time * a,
                              kTimeSteps * kTolerance));
  EXPECT_TRUE(CompareMatrices(state_n.get_position(),
                              state_0.get_position() +
                                  total_time * state_0.get_velocity() +
                                  0.5 * total_time * total_time * a,
                              kTimeSteps * kTolerance));

  /* If `state->CopyFrom(prev_state);` is not called within
   AccelerationNewmarkScheme::AdvanceDt(), the following EXPECT_EQs
   will fail. */
  DerState<double> state_n_expected(der_state_system_.get());
  state_n_expected.SetAcceleration(a);
  DerState<double> state_np1(der_state_system_.get());
  for (int i = 0; i < kTimeSteps; ++i) {
    LimitMalloc guard;
    scheme_.AdvanceDt(state_n_expected, a, &state_np1);
    state_n_expected.CopyFrom(state_np1);
  }
  EXPECT_EQ(state_n.get_reference_frame_d1(),
            state_n_expected.get_reference_frame_d1());
  EXPECT_EQ(state_n.get_twist(), state_n_expected.get_twist());
}

/* Verify that the result of AdjustStateFromChangeInUnknowns() is consistent
 with the weights. */
TEST_P(AccelerationNewmarkSchemeTest, AdjustStateFromChangeInUnknowns) {
  const DerState<double> state0(der_state_system_.get());
  DerState<double> state(der_state_system_.get());
  const VectorXd dz = VectorXd::LinSpaced(state.num_dofs(), 0.0, 1.0);
  const std::array<double, 3> weights = scheme_.GetWeights();
  {
    LimitMalloc guard;
    scheme_.AdjustStateFromChangeInUnknowns(dz, &state);
  }

  EXPECT_TRUE(CompareMatrices(state.get_position() - state0.get_position(),
                              weights[0] * dz, kTolerance));
  EXPECT_TRUE(CompareMatrices(state.get_velocity() - state0.get_velocity(),
                              weights[1] * dz, kTolerance));
  EXPECT_TRUE(
      CompareMatrices(state.get_acceleration() - state0.get_acceleration(),
                      weights[2] * dz, kTolerance));
}

/* Tests that `AccelerationNewmarkScheme` is equivalent to
 `VelocityNewmarkScheme` by advancing one time step with each integration
 scheme using the same initial state and verifying that the resulting new
 states are the same. */
TEST_P(AccelerationNewmarkSchemeTest, EquivalenceWithVelocityNewmark) {
  const DerState<double> state0(der_state_system_.get());
  DerState<double> state_a(der_state_system_.get());
  const VectorXd a = VectorXd::LinSpaced(state0.num_dofs(), 0.0, 1.0);
  scheme_.AdvanceDt(state0, a, &state_a);

  const VelocityNewmarkScheme<double> velocity_scheme{kDt, kGamma, kBeta};
  DerState<double> state_v(der_state_system_.get());
  velocity_scheme.AdvanceDt(state0, state_a.get_velocity(), &state_v);

  /* Set a larger error tolerance to accommodate the division by `dt` used in
   the VelocityNewmarkScheme. */
  EXPECT_TRUE(CompareMatrices(state_v.get_acceleration(),
                              state_a.get_acceleration(), kTolerance / kDt));
  EXPECT_TRUE(CompareMatrices(state_v.get_velocity(), state_a.get_velocity(),
                              kTolerance));
  EXPECT_TRUE(CompareMatrices(state_v.get_position(), state_a.get_position(),
                              kTolerance));
}

}  // namespace
}  // namespace internal
}  // namespace der
}  // namespace multibody
}  // namespace drake
