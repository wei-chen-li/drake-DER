#include "drake/multibody/der/der_undeformed_state.h"

#include <algorithm>

#include "drake/solvers/mathematical_program.h"
#include "drake/solvers/solve.h"

namespace drake {
namespace multibody {
namespace der {

namespace {

/* Returns the angle each edge occupies in a circle. */
Eigen::RowVectorXd AngleOfEdgeInCircle(
    const Eigen::Ref<const Eigen::RowVectorXd>& edge_length) {
  const int num_edges = edge_length.size();
  DRAKE_THROW_UNLESS(num_edges >= 2);

  solvers::MathematicalProgram prog;
  auto half_angle = prog.NewContinuousVariables(num_edges, "half_angle");
  auto r = prog.NewContinuousVariables(1, "r")[0];
  for (int i = 0; i < num_edges; ++i) {
    prog.AddConstraint(r * sin(half_angle[i]) == edge_length[i] / 2.0);
  }
  prog.AddLinearEqualityConstraint(Eigen::RowVectorXd::Ones(num_edges), M_PI,
                                   half_angle);

  Eigen::VectorXd initial_guess(num_edges + 1);
  const double total_length =
      std::accumulate(edge_length.begin(), edge_length.end(), 0.0);
  for (int i = 0; i < num_edges; ++i) {
    initial_guess[i] = edge_length[i] / total_length * M_PI;
  }
  initial_guess[num_edges] = total_length / (2 * M_PI);
  solvers::MathematicalProgramResult result =
      solvers::Solve(prog, initial_guess);
  DRAKE_DEMAND(result.is_success());
  return result.GetSolution(half_angle) * 2.0;
}

}  // namespace

template <typename T>
DerUndeformedState<T>& DerUndeformedState<T>::operator=(
    const DerUndeformedState<T>& other) {
  DRAKE_THROW_UNLESS(this->has_closed_ends() == other.has_closed_ends());
  DRAKE_THROW_UNLESS(this->num_nodes() == other.num_nodes());
  this->edge_length_ = other.edge_length_;
  this->voronoi_length_ = other.voronoi_length_;
  this->kappa1_ = other.kappa1_;
  this->kappa2_ = other.kappa2_;
  this->twist_ = other.twist_;
  return *this;
}

template <typename T>
DerUndeformedState<T> DerUndeformedState<T>::ZeroCurvatureAndTwist(
    bool has_closed_ends,
    const Eigen::Ref<const Eigen::RowVectorX<T>>& edge_length) {
  const int num_edges = edge_length.size();
  const int num_internal_nodes = has_closed_ends ? num_edges : num_edges - 1;
  DRAKE_THROW_UNLESS(num_edges > 0);
  DRAKE_THROW_UNLESS((edge_length.array() > 0).all());
  auto zero = Eigen::RowVectorX<T>::Zero(num_internal_nodes);
  return DerUndeformedState<T>(has_closed_ends, edge_length,
                               /* kappa1 = */ zero, /* kappa2 = */ zero,
                               /* twist = */ zero);
}

template <typename T>
DerUndeformedState<T> DerUndeformedState<T>::NaturalCurvatureZeroTwist(
    const internal::DerState<T>& state) {
  const int num_internal_nodes = state.num_internal_nodes();
  Eigen::RowVectorX<T> kappa1(num_internal_nodes);
  Eigen::RowVectorX<T> kappa2(num_internal_nodes);
  if (!state.has_closed_ends()) {
    /* If the DER has open ends, set the curvature to that of a straight line.
     */
    kappa1 = Eigen::RowVectorX<T>::Zero(num_internal_nodes);
    kappa2 = Eigen::RowVectorX<T>::Zero(num_internal_nodes);
  } else {
    /* If the DER has open ends, set the curvature to that of a circle. */
    /* Get the angle each edge occupies in a circle. */
    const Eigen::RowVectorXd edge_angle =
        AngleOfEdgeInCircle(ExtractDoubleOrThrow(state.get_edge_length()));
    const Eigen::Matrix3Xd t = ExtractDoubleOrThrow(state.get_tangent());
    /* Get the normal of the plane the DER is on. */
    constexpr double kTol = 1e-14;
    Eigen::Vector3d plane_normal = Eigen::Vector3d::Zero();
    for (int i = 0; i < state.num_internal_nodes(); ++i) {
      const int ip1 = (i + 1) % state.num_edges();
      const Eigen::Vector3d b = t.col(i).cross(t.col(ip1));
      if (b.norm() >= kTol) plane_normal += b.normalized();
    }
    plane_normal = (plane_normal.norm() >= kTol) ? plane_normal.normalized()
                                                 : Eigen::Vector3d(1, 0, 0);
    /* Get the discrete integrated curvature and project onto d₁ and d₂. */
    const auto& d1 = state.get_reference_frame_d1();
    const auto& d2 = state.get_reference_frame_d2();
    for (int i = 0; i < num_internal_nodes; ++i) {
      const int ip1 = (i + 1) % state.num_edges();
      const double kappa = 2.0 * tan(edge_angle[ip1] / 2.0);
      const Eigen::Vector3d b = t.col(i).cross(t.col(ip1));
      const Eigen::Vector3d kappa_b =
          (b.norm() >= kTol) ? kappa * b.normalized() : kappa * plane_normal;

      kappa1[i] = 0.5 * (d2.col(i) + d2.col(ip1)).dot(kappa_b);
      kappa2[i] = -0.5 * (d1.col(i) + d1.col(ip1)).dot(kappa_b);
    }
  }
  const auto twist = Eigen::RowVectorX<T>::Zero(num_internal_nodes);
  return DerUndeformedState<T>(state.has_closed_ends(), state.get_edge_length(),
                               std::move(kappa1), std::move(kappa2), twist);
}

template <typename T>
void DerUndeformedState<T>::set_edge_length(
    const Eigen::Ref<const Eigen::RowVectorX<T>>& edge_length) {
  DRAKE_THROW_UNLESS(edge_length.size() == num_edges());
  DRAKE_THROW_UNLESS((edge_length.array() > 0).all());
  edge_length_ = edge_length;
  for (int i = 0; i < num_internal_nodes(); ++i) {
    const int ip1 = (i + 1) % num_edges();
    voronoi_length_[i] = 0.5 * (edge_length_[i] + edge_length_[ip1]);
  }
}

template <typename T>
void DerUndeformedState<T>::set_curvature_kappa(
    const Eigen::Ref<const Eigen::RowVectorX<T>>& kappa1,
    const Eigen::Ref<const Eigen::RowVectorX<T>>& kappa2) {
  DRAKE_THROW_UNLESS(kappa1.size() == num_internal_nodes());
  DRAKE_THROW_UNLESS(kappa2.size() == num_internal_nodes());
  kappa1_ = kappa1;
  kappa2_ = kappa2;
}

template <typename T>
void DerUndeformedState<T>::set_curvature_angle(
    const Eigen::Ref<const Eigen::RowVectorX<T>>& angle1,
    const Eigen::Ref<const Eigen::RowVectorX<T>>& angle2) {
  DRAKE_THROW_UNLESS(angle1.size() == num_internal_nodes());
  DRAKE_THROW_UNLESS(angle1.size() == num_internal_nodes());
  DRAKE_THROW_UNLESS((angle1.array() > -2 * M_PI).all());
  DRAKE_THROW_UNLESS((angle1.array() < 2 * M_PI).all());
  DRAKE_THROW_UNLESS((angle2.array() > -2 * M_PI).all());
  DRAKE_THROW_UNLESS((angle2.array() < 2 * M_PI).all());
  kappa1_ = 2.0 * tan(angle1.array() / 2.0);
  kappa2_ = 2.0 * tan(angle2.array() / 2.0);
}

template <typename T>
void DerUndeformedState<T>::set_twist(
    const Eigen::Ref<const Eigen::RowVectorX<T>>& twist) {
  DRAKE_THROW_UNLESS(twist.size() == num_internal_nodes());
  twist_ = twist;
}

template <typename T>
DerUndeformedState<T>::DerUndeformedState(bool has_closed_ends,
                                          Eigen::RowVectorX<T> edge_length,
                                          Eigen::RowVectorX<T> kappa1,
                                          Eigen::RowVectorX<T> kappa2,
                                          Eigen::RowVectorX<T> twist)
    : has_closed_ends_(has_closed_ends),
      edge_length_(std::move(edge_length)),
      kappa1_(std::move(kappa1)),
      kappa2_(std::move(kappa2)),
      twist_(std::move(twist)) {
  DRAKE_THROW_UNLESS(num_edges() ==
                     (has_closed_ends_ ? num_nodes() : num_nodes() - 1));
  DRAKE_THROW_UNLESS(num_internal_nodes() ==
                     (has_closed_ends_ ? num_nodes() : num_nodes() - 2));
  DRAKE_THROW_UNLESS(edge_length_.size() == num_edges());
  DRAKE_THROW_UNLESS(kappa1_.size() == num_internal_nodes());
  DRAKE_THROW_UNLESS(kappa2_.size() == num_internal_nodes());
  DRAKE_THROW_UNLESS(twist_.size() == num_internal_nodes());

  voronoi_length_.resize(num_internal_nodes());
  for (int i = 0; i < num_internal_nodes(); ++i) {
    const int ip1 = (i + 1) % num_edges();
    voronoi_length_[i] = 0.5 * (edge_length_[i] + edge_length_[ip1]);
  }
}

template <typename T>
template <typename U>
DerUndeformedState<U> DerUndeformedState<T>::ToScalarType() const {
  static_assert(!std::is_same_v<T, U>);
  return DerUndeformedState<U>(
      has_closed_ends_, ExtractDoubleOrThrow(edge_length_),
      ExtractDoubleOrThrow(kappa1_), ExtractDoubleOrThrow(kappa2_),
      ExtractDoubleOrThrow(twist_));
}

using symbolic::Expression;
template DerUndeformedState<AutoDiffXd>
DerUndeformedState<double>::ToScalarType<AutoDiffXd>() const;
template DerUndeformedState<Expression>
DerUndeformedState<double>::ToScalarType<Expression>() const;
template DerUndeformedState<double>
DerUndeformedState<AutoDiffXd>::ToScalarType<double>() const;
template DerUndeformedState<Expression>
DerUndeformedState<AutoDiffXd>::ToScalarType<Expression>() const;
template DerUndeformedState<double>
DerUndeformedState<Expression>::ToScalarType<double>() const;
template DerUndeformedState<AutoDiffXd>
DerUndeformedState<Expression>::ToScalarType<AutoDiffXd>() const;

}  // namespace der
}  // namespace multibody
}  // namespace drake

DRAKE_DEFINE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_SCALARS(
    class ::drake::multibody::der::DerUndeformedState);
