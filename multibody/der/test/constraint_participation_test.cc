#include "drake/multibody/der/constraint_participation.h"

#include <algorithm>
#include <unordered_set>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace drake {
namespace multibody {
namespace der {
namespace internal {
namespace {

using multibody::contact_solvers::internal::PartialPermutation;
using testing::UnorderedElementsAreArray;

class ConstraintParticipationTest : public ::testing::TestWithParam<bool> {
 private:
  void SetUp() override { EXPECT_TRUE(constraint_participation_.empty()); }

 protected:
  const bool has_closed_ends_ = GetParam();
  const int num_edges_ = 4;
  const int num_nodes_ = has_closed_ends_ ? num_edges_ : num_edges_ + 1;
  const int num_dofs_ = num_nodes_ * 3 + num_edges_;
  ConstraintParticipation constraint_participation_{num_dofs_};
};

INSTANTIATE_TEST_SUITE_P(HasClosedEnds, ConstraintParticipationTest,
                         ::testing::Values(false, true));

[[nodiscard]] static bool ParticipatingDofsMatch(
    const PartialPermutation& dof_permutation,
    const std::unordered_set<int>& participating_dofs) {
  const int num_dofs = dof_permutation.domain_size();
  int permuted_index = 0;
  for (int i = 0; i < num_dofs; ++i) {
    if (dof_permutation.participates(i) != participating_dofs.contains(i))
      return false;

    if (dof_permutation.participates(i)) {
      if (dof_permutation.permuted_index(i) != permuted_index++) return false;
    }
  }
  return true;
}

TEST_P(ConstraintParticipationTest, ParticipateNodes) {
  constraint_participation_.ParticipateNodes({0, 1});
  EXPECT_FALSE(constraint_participation_.empty());

  std::unordered_set<int> participating_dofs = {0, 1, 2, 4, 5, 6};
  EXPECT_THAT(constraint_participation_.ComputeParticipatingDofs(),
              UnorderedElementsAreArray(participating_dofs));

  PartialPermutation dof_permutation =
      constraint_participation_.ComputeDofPermutation();
  EXPECT_EQ(dof_permutation.domain_size(), num_dofs_);
  EXPECT_TRUE(ParticipatingDofsMatch(dof_permutation, participating_dofs));
}

TEST_P(ConstraintParticipationTest, ParticipateEdges) {
  constraint_participation_.ParticipateEdges({1, 3});
  EXPECT_FALSE(constraint_participation_.empty());

  std::unordered_set<int> participating_dofs = {7, 15};
  EXPECT_THAT(constraint_participation_.ComputeParticipatingDofs(),
              UnorderedElementsAreArray(participating_dofs));

  PartialPermutation dof_permutation =
      constraint_participation_.ComputeDofPermutation();
  EXPECT_EQ(dof_permutation.domain_size(), num_dofs_);
  EXPECT_TRUE(ParticipatingDofsMatch(dof_permutation, participating_dofs));
}

TEST_P(ConstraintParticipationTest, ParticipateEdgesAndAdjacentNodes) {
  constraint_participation_.ParticipateEdgesAndAdjacentNodes({1, 3});
  EXPECT_FALSE(constraint_participation_.empty());

  std::unordered_set<int> participating_dofs;
  if (!has_closed_ends_) {
    participating_dofs.insert({4, 5, 6, 7, 8, 9, 10});
    participating_dofs.insert({12, 13, 14, 15, 16, 17, 18});
  } else {
    participating_dofs.insert({4, 5, 6, 7, 8, 9, 10});
    participating_dofs.insert({12, 13, 14, 15, 0, 1, 2});
  }
  EXPECT_THAT(constraint_participation_.ComputeParticipatingDofs(),
              UnorderedElementsAreArray(participating_dofs));

  PartialPermutation dof_permutation =
      constraint_participation_.ComputeDofPermutation();
  EXPECT_EQ(dof_permutation.domain_size(), num_dofs_);
  EXPECT_TRUE(ParticipatingDofsMatch(dof_permutation, participating_dofs));
}

}  // namespace
}  // namespace internal
}  // namespace der
}  // namespace multibody
}  // namespace drake
