#include "drake/multibody/der/dirichlet_boundary_condition.h"

#include <memory>
#include <vector>

#include <gtest/gtest.h>

#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/common/test_utilities/expect_throws_message.h"
#include "drake/common/test_utilities/limit_malloc.h"
#include "drake/common/text_logging.h"
#include "drake/multibody/der/elastic_energy.h"

namespace drake {
namespace multibody {
namespace der {
namespace internal {
namespace {

using Eigen::Matrix3d;
using Eigen::MatrixXd;
using Eigen::Vector3d;
using Eigen::VectorXd;
using test::LimitMalloc;

class DirichletBoundaryConditionTest : public ::testing::Test {
 protected:
  void SetUp() {
    bool has_closed_ends = true;
    std::vector<Vector3d> node_positions = {
        Vector3d(0, 0, 0), Vector3d(1, 0, 0), Vector3d(2, 0, 0)};
    std::vector<double> edge_angles = {4, 5, 6};
    der_state_system_ = std::make_unique<DerStateSystem<double>>(
        has_closed_ends, node_positions, edge_angles, std::nullopt);
    EXPECT_EQ(der_state_system_->num_dofs(), kDofs);

    state_ = std::make_unique<DerState<double>>(der_state_system_.get());

    node_state_ = NodeState<double>{.x = Vector3d::Constant(0.1),
                                    .x_dot = Vector3d::Constant(0.2),
                                    .x_ddot = Vector3d::Constant(0.3)};
    bc_.AddBoundaryCondition(DerNodeIndex(0), node_state_);

    edge_state_ =
        EdgeState<double>{.gamma = 0.4, .gamma_dot = 0.5, .gamma_ddot = 0.6};
    bc_.AddBoundaryCondition(DerEdgeIndex(1), edge_state_);
  }

  EnergyHessianMatrix<double> MakeTangentMatrix() const {
    EnergyHessianMatrix<double> mat =
        EnergyHessianMatrix<double>::Allocate(kDofs);
    EXPECT_EQ(mat.rows(), kDofs);
    for (int i = 0; i < 3; ++i) {
      for (int j = i; j < 3; ++j) {
        const double val = (i != j) ? (i + j) : i + 4.0;
        mat.Insert(DerNodeIndex(i), DerNodeIndex(j), Matrix3d::Constant(val));
        mat.Insert(DerNodeIndex(i), DerEdgeIndex(j), Vector3d::Constant(val));
        mat.Insert(DerEdgeIndex(i), DerEdgeIndex(j), val);
        if (i == j) continue;
        mat.Insert(DerNodeIndex(j), DerEdgeIndex(i), Vector3d::Constant(val));
      }
    }

    MatrixXd expected_mat(kDofs, kDofs);
    // clang-format off
    expected_mat << 4, 4, 4, 4,   1, 1, 1, 1,   2, 2, 2, 2,
                    4, 4, 4, 4,   1, 1, 1, 1,   2, 2, 2, 2,
                    4, 4, 4, 4,   1, 1, 1, 1,   2, 2, 2, 2,
                    4, 4, 4, 4,   1, 1, 1, 1,   2, 2, 2, 2,

                    1, 1, 1, 1,   5, 5, 5, 5,   3, 3, 3, 3,
                    1, 1, 1, 1,   5, 5, 5, 5,   3, 3, 3, 3,
                    1, 1, 1, 1,   5, 5, 5, 5,   3, 3, 3, 3,
                    1, 1, 1, 1,   5, 5, 5, 5,   3, 3, 3, 3,

                    2, 2, 2, 2,   3, 3, 3, 3,   6, 6, 6, 6,
                    2, 2, 2, 2,   3, 3, 3, 3,   6, 6, 6, 6,
                    2, 2, 2, 2,   3, 3, 3, 3,   6, 6, 6, 6,
                    2, 2, 2, 2,   3, 3, 3, 3,   6, 6, 6, 6;
    // clang-format on
    EXPECT_TRUE(CompareMatrices(mat.MakeDenseMatrix(), expected_mat));
    return mat;
  }

  static constexpr int kDofs = 12;
  std::unique_ptr<DerStateSystem<double>> der_state_system_;
  std::unique_ptr<DerState<double>> state_;
  DirichletBoundaryCondition<double> bc_;
  NodeState<double> node_state_;
  EdgeState<double> edge_state_;
};

TEST_F(DirichletBoundaryConditionTest, GetBoundaryCondition) {
  EXPECT_NE(bc_.GetBoundaryCondition(DerNodeIndex(0)), nullptr);
  EXPECT_EQ(bc_.GetBoundaryCondition(DerNodeIndex(0))->x, node_state_.x);
  EXPECT_EQ(bc_.GetBoundaryCondition(DerNodeIndex(0))->x_dot,
            node_state_.x_dot);
  EXPECT_EQ(bc_.GetBoundaryCondition(DerNodeIndex(0))->x_ddot,
            node_state_.x_ddot);

  EXPECT_EQ(bc_.GetBoundaryCondition(DerNodeIndex(1)), nullptr);

  EXPECT_NE(bc_.GetBoundaryCondition(DerEdgeIndex(1)), nullptr);
  EXPECT_EQ(bc_.GetBoundaryCondition(DerEdgeIndex(1))->gamma,
            edge_state_.gamma);
  EXPECT_EQ(bc_.GetBoundaryCondition(DerEdgeIndex(1))->gamma_dot,
            edge_state_.gamma_dot);
  EXPECT_EQ(bc_.GetBoundaryCondition(DerEdgeIndex(1))->gamma_ddot,
            edge_state_.gamma_ddot);

  EXPECT_EQ(bc_.GetBoundaryCondition(DerEdgeIndex(2)), nullptr);
}

/* Tests that the DirichletBoundaryCondition under test successfully modifies
 a given state. */
TEST_F(DirichletBoundaryConditionTest, ApplyBoundaryConditionToState) {
  {
    LimitMalloc guard;
    bc_.ApplyBoundaryConditionToState(state_.get());
  }
  VectorXd expected_q(kDofs), expected_v(kDofs), expected_a(kDofs);
  expected_q << 0.1, 0.1, 0.1, 4, 1, 0, 0, 0.4, 2, 0, 0, 6;
  expected_v << 0.2, 0.2, 0.2, 0, 0, 0, 0, 0.5, 0, 0, 0, 0;
  expected_a << 0.3, 0.3, 0.3, 0, 0, 0, 0, 0.6, 0, 0, 0, 0;
  EXPECT_TRUE(CompareMatrices(state_->get_position(), expected_q));
  EXPECT_TRUE(CompareMatrices(state_->get_velocity(), expected_v));
  EXPECT_TRUE(CompareMatrices(state_->get_acceleration(), expected_a));
}

/* Tests that the DirichletBoundaryCondition under test successfully modifies
 a given residual. */
TEST_F(DirichletBoundaryConditionTest, ApplyHomogeneousBoundaryCondition) {
  VectorXd residual(kDofs);
  residual << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12;
  {
    LimitMalloc guard;
    bc_.ApplyHomogeneousBoundaryCondition(&residual);
  }
  VectorXd expected_residual(kDofs);
  expected_residual << 0, 0, 0, 4, 5, 6, 7, 0, 9, 10, 11, 12;
  EXPECT_TRUE(CompareMatrices(residual, expected_residual));
}

/* Tests that the DirichletBoundaryCondition under test successfully modifies a
 given tangent matrix. */
TEST_F(DirichletBoundaryConditionTest, ApplyBoundaryConditionToTangentMatrix) {
  EnergyHessianMatrix<double> tangent_matrix = MakeTangentMatrix();
  drake::log()->info("\n{}", fmt_eigen(tangent_matrix.MakeDenseMatrix()));
  {
    LimitMalloc guard;
    bc_.ApplyBoundaryConditionToTangentMatrix(&tangent_matrix);
  }

  MatrixXd expected_tangent_matrix(kDofs, kDofs);
  // clang-format off
  expected_tangent_matrix << 1, 0, 0, 0,   0, 0, 0, 0,   0, 0, 0, 0,
                             0, 1, 0, 0,   0, 0, 0, 0,   0, 0, 0, 0,
                             0, 0, 1, 0,   0, 0, 0, 0,   0, 0, 0, 0,
                             0, 0, 0, 4,   1, 1, 1, 0,   2, 2, 2, 2,

                             0, 0, 0, 1,   5, 5, 5, 0,   3, 3, 3, 3,
                             0, 0, 0, 1,   5, 5, 5, 0,   3, 3, 3, 3,
                             0, 0, 0, 1,   5, 5, 5, 0,   3, 3, 3, 3,
                             0, 0, 0, 0,   0, 0, 0, 1,   0, 0, 0, 0,

                             0, 0, 0, 2,   3, 3, 3, 0,   6, 6, 6, 6,
                             0, 0, 0, 2,   3, 3, 3, 0,   6, 6, 6, 6,
                             0, 0, 0, 2,   3, 3, 3, 0,   6, 6, 6, 6,
                             0, 0, 0, 2,   3, 3, 3, 0,   6, 6, 6, 6;
  // clang-format on

  EXPECT_TRUE(CompareMatrices(tangent_matrix.MakeDenseMatrix(),
                              expected_tangent_matrix));
}

/* Tests out-of-bound node boundary condition throw an exception. */
TEST_F(DirichletBoundaryConditionTest, NodeOutOfBound) {
  bc_.AddBoundaryCondition(
      DerNodeIndex(state_->num_nodes()),
      {Vector3d::Zero(), Vector3d::Zero(), Vector3d::Zero()});
  DRAKE_EXPECT_THROWS_MESSAGE(
      bc_.ApplyBoundaryConditionToState(state_.get()),
      "A node index of the Dirichlet boundary condition is out of range.");
  VectorXd residual(kDofs);
  DRAKE_EXPECT_THROWS_MESSAGE(
      bc_.ApplyHomogeneousBoundaryCondition(&residual),
      "A node index of the Dirichlet boundary condition is out of range.");
  EnergyHessianMatrix<double> tangent_matrix = MakeTangentMatrix();
  DRAKE_EXPECT_THROWS_MESSAGE(
      bc_.ApplyBoundaryConditionToTangentMatrix(&tangent_matrix),
      "A node index of the Dirichlet boundary condition is out of range.");
}

/* Tests out-of-bound node boundary condition throw an exception. */
TEST_F(DirichletBoundaryConditionTest, EdgeOutOfBound) {
  bc_.AddBoundaryCondition(DerEdgeIndex(state_->num_edges()), {0.0, 0.0, 0.0});
  DRAKE_EXPECT_THROWS_MESSAGE(
      bc_.ApplyBoundaryConditionToState(state_.get()),
      "An edge index of the Dirichlet boundary condition is out of range.");
  VectorXd residual(kDofs);
  DRAKE_EXPECT_THROWS_MESSAGE(
      bc_.ApplyHomogeneousBoundaryCondition(&residual),
      "An edge index of the Dirichlet boundary condition is out of range.");
  EnergyHessianMatrix<double> tangent_matrix = MakeTangentMatrix();
  DRAKE_EXPECT_THROWS_MESSAGE(
      bc_.ApplyBoundaryConditionToTangentMatrix(&tangent_matrix),
      "An edge index of the Dirichlet boundary condition is out of range.");
}

}  // namespace
}  // namespace internal
}  // namespace der
}  // namespace multibody
}  // namespace drake
