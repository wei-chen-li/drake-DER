#include "drake/multibody/der/der_undeformed_state.h"

#include <vector>

#include <gtest/gtest.h>

#include "drake/common/autodiff.h"
#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/multibody/der/der_state.h"

namespace drake {
namespace multibody {
namespace der {
namespace {

using Eigen::RowVectorXd;
using Eigen::Vector3d;

class DerUndeformedStateTest : public ::testing::TestWithParam<bool> {};

INSTANTIATE_TEST_SUITE_P(HasClosedEnds, DerUndeformedStateTest,
                         ::testing::Values(false, true));

TEST_P(DerUndeformedStateTest, CopyAssignmentOperator) {
  const bool has_closed_ends = GetParam();

  RowVectorXd edge_length(3);
  edge_length << 0.10, 0.12, 0.08;
  const auto a = DerUndeformedState<double>::ZeroCurvatureAndTwist(
      has_closed_ends, RowVectorXd::Ones(3));

  auto b = DerUndeformedState<double>::ZeroCurvatureAndTwist(
      has_closed_ends, RowVectorXd::Ones(3));
  EXPECT_NO_THROW(b = a);
  EXPECT_EQ(b.get_edge_length(), a.get_edge_length());
  EXPECT_EQ(b.get_voronoi_length(), a.get_voronoi_length());
  EXPECT_EQ(b.get_curvature_kappa1(), a.get_curvature_kappa1());
  EXPECT_EQ(b.get_curvature_kappa2(), a.get_curvature_kappa2());
  EXPECT_EQ(b.get_twist(), a.get_twist());

  auto c = DerUndeformedState<double>::ZeroCurvatureAndTwist(
      has_closed_ends, RowVectorXd::Ones(2));
  EXPECT_ANY_THROW(c = a);
}

TEST_P(DerUndeformedStateTest, ZeroCurvatureAndTwist) {
  const bool has_closed_ends = GetParam();

  const int num_edges = 3;
  RowVectorXd edge_length(num_edges);
  edge_length << 0.10, 0.12, 0.08;
  const auto undeformed = DerUndeformedState<double>::ZeroCurvatureAndTwist(
      has_closed_ends, edge_length);
  EXPECT_EQ(undeformed.has_closed_ends(), has_closed_ends);

  EXPECT_TRUE(CompareMatrices(undeformed.get_edge_length(), edge_length));

  const int num_internal_nodes = has_closed_ends ? num_edges : num_edges - 1;
  RowVectorXd voronoi_length(num_internal_nodes);
  if (has_closed_ends) {
    voronoi_length << 0.11, 0.10, 0.09;
  } else {
    voronoi_length << 0.11, 0.10;
  }
  EXPECT_TRUE(CompareMatrices(undeformed.get_voronoi_length(), voronoi_length));

  const auto zero = RowVectorXd::Zero(num_internal_nodes);
  EXPECT_TRUE(CompareMatrices(undeformed.get_curvature_kappa1(), zero));
  EXPECT_TRUE(CompareMatrices(undeformed.get_curvature_kappa2(), zero));
  EXPECT_TRUE(CompareMatrices(undeformed.get_twist(), zero));
}

TEST_P(DerUndeformedStateTest, NaturalCurvatureZeroTwist) {
  const bool has_closed_ends = GetParam();
  const int num_nodes = 8;
  const int num_edges = has_closed_ends ? num_nodes : num_nodes - 1;
  const int num_internal_nodes = has_closed_ends ? num_nodes : num_nodes - 2;

  const double a = 1.0;
  std::vector<Vector3d> node_positions = {
      Vector3d(-a, -a, 0), Vector3d(0, -a, 0), Vector3d(a, -a, 0),
      Vector3d(a, 0, 0),   Vector3d(a, a, 0),  Vector3d(0, a, 0),
      Vector3d(-a, a, 0),  Vector3d(-a, 0, 0),
  };
  std::vector<double> edge_angles(num_edges, 0.0);
  Vector3d d1_0(0, 1.0 / 2, sqrt(3) / 2);

  const internal::DerStateSystem<double> der_state_system(
      has_closed_ends, node_positions, edge_angles, d1_0);
  const internal::DerState<double> der_state(&der_state_system);
  const DerUndeformedState<double> undeformed =
      DerUndeformedState<double>::NaturalCurvatureZeroTwist(der_state);

  if (!has_closed_ends) {
    EXPECT_TRUE(CompareMatrices(undeformed.get_edge_length(),
                                RowVectorXd::Constant(num_edges, a)));
    const auto zero = RowVectorXd::Zero(num_internal_nodes);
    EXPECT_TRUE(CompareMatrices(undeformed.get_curvature_kappa1(), zero));
    EXPECT_TRUE(CompareMatrices(undeformed.get_curvature_kappa2(), zero));
    EXPECT_TRUE(CompareMatrices(undeformed.get_twist(), zero));
  } else {
    EXPECT_TRUE(CompareMatrices(undeformed.get_edge_length(),
                                RowVectorXd::Constant(num_edges, a)));
    EXPECT_TRUE(CompareMatrices(undeformed.get_twist(),
                                RowVectorXd::Zero(num_internal_nodes)));

    const double angle = (2 * M_PI) / num_edges;
    const double kappa = 2 * tan(angle / 2);
    constexpr double kTol = 1e-15;
    EXPECT_TRUE(CompareMatrices(
        undeformed.get_curvature_kappa1(),
        RowVectorXd::Constant(num_edges, kappa * (1.0 / 2)), kTol));
    EXPECT_TRUE(CompareMatrices(
        undeformed.get_curvature_kappa2(),
        RowVectorXd::Constant(num_edges, kappa * (-sqrt(3) / 2)), kTol));
  }
}

TEST_P(DerUndeformedStateTest, set_edge_length) {
  const bool has_closed_ends = GetParam();

  const int num_edges = 3;
  RowVectorXd edge_length = RowVectorXd::Constant(num_edges, 0.1);
  auto undeformed = DerUndeformedState<double>::ZeroCurvatureAndTwist(
      has_closed_ends, edge_length);

  edge_length << 0.10, 0.12, 0.08;
  undeformed.set_edge_length(edge_length);
  EXPECT_TRUE(CompareMatrices(undeformed.get_edge_length(), edge_length));

  const int num_internal_nodes = has_closed_ends ? num_edges : num_edges - 1;
  RowVectorXd voronoi_length(num_internal_nodes);
  if (has_closed_ends) {
    voronoi_length << 0.11, 0.10, 0.09;
  } else {
    voronoi_length << 0.11, 0.10;
  }
  EXPECT_TRUE(CompareMatrices(undeformed.get_voronoi_length(), voronoi_length));

  const auto zero = RowVectorXd::Zero(num_internal_nodes);
  EXPECT_TRUE(CompareMatrices(undeformed.get_curvature_kappa1(), zero));
  EXPECT_TRUE(CompareMatrices(undeformed.get_curvature_kappa2(), zero));
  EXPECT_TRUE(CompareMatrices(undeformed.get_twist(), zero));
}

TEST_P(DerUndeformedStateTest, set_curvature_kappa) {
  const bool has_closed_ends = GetParam();

  const int num_edges = 3;
  const auto edge_length = RowVectorXd::Constant(num_edges, 0.1);
  auto undeformed = DerUndeformedState<double>::ZeroCurvatureAndTwist(
      has_closed_ends, edge_length);

  const int num_internal_nodes = has_closed_ends ? num_edges : num_edges - 1;
  const auto kappa1 = RowVectorXd::LinSpaced(num_internal_nodes, 0.0, 1.0);
  const auto kappa2 = RowVectorXd::LinSpaced(num_internal_nodes, 1.0, 2.0);
  undeformed.set_curvature_kappa(kappa1, kappa2);

  EXPECT_TRUE(CompareMatrices(undeformed.get_curvature_kappa1(), kappa1));
  EXPECT_TRUE(CompareMatrices(undeformed.get_curvature_kappa2(), kappa2));
  EXPECT_TRUE(CompareMatrices(undeformed.get_edge_length(), edge_length));
  const auto zero = RowVectorXd::Zero(num_internal_nodes);
  EXPECT_TRUE(CompareMatrices(undeformed.get_twist(), zero));
}

TEST_P(DerUndeformedStateTest, set_curvature_angle) {
  const bool has_closed_ends = GetParam();

  const int num_edges = 3;
  const auto edge_length = RowVectorXd::Constant(num_edges, 0.1);
  auto undeformed = DerUndeformedState<double>::ZeroCurvatureAndTwist(
      has_closed_ends, edge_length);

  const int num_internal_nodes = has_closed_ends ? num_edges : num_edges - 1;
  const auto angle1 =
      RowVectorXd::LinSpaced(num_internal_nodes, 0.0, M_PI * 0.9);
  const auto angle2 =
      RowVectorXd::LinSpaced(num_internal_nodes, -M_PI * 0.9, 0.0);
  undeformed.set_curvature_angle(angle1, angle2);

  EXPECT_TRUE(CompareMatrices(undeformed.get_curvature_kappa1(),
                              RowVectorXd(2 * tan(angle1.array() / 2))));
  EXPECT_TRUE(CompareMatrices(undeformed.get_curvature_kappa2(),
                              RowVectorXd(2 * tan(angle2.array() / 2))));
  EXPECT_TRUE(CompareMatrices(undeformed.get_edge_length(), edge_length));
  const auto zero = RowVectorXd::Zero(num_internal_nodes);
  EXPECT_TRUE(CompareMatrices(undeformed.get_twist(), zero));
}

TEST_P(DerUndeformedStateTest, set_twist) {
  const bool has_closed_ends = GetParam();

  const int num_edges = 3;
  const auto edge_length = RowVectorXd::Constant(num_edges, 0.1);
  auto undeformed = DerUndeformedState<double>::ZeroCurvatureAndTwist(
      has_closed_ends, edge_length);

  const int num_internal_nodes = has_closed_ends ? num_edges : num_edges - 1;
  const auto twist = RowVectorXd::LinSpaced(num_internal_nodes, 0.0, 1.0);
  undeformed.set_twist(twist);

  EXPECT_TRUE(CompareMatrices(undeformed.get_twist(), twist));
  EXPECT_TRUE(CompareMatrices(undeformed.get_edge_length(), edge_length));
  const auto zero = RowVectorXd::Zero(num_internal_nodes);
  EXPECT_TRUE(CompareMatrices(undeformed.get_curvature_kappa1(), zero));
  EXPECT_TRUE(CompareMatrices(undeformed.get_curvature_kappa2(), zero));
}

}  // namespace
}  // namespace der
}  // namespace multibody
}  // namespace drake
