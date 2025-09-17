#include "drake/multibody/der/constraint_participation.h"

#include <utility>
#include <vector>

namespace drake {
namespace multibody {
namespace der {
namespace internal {

ConstraintParticipation::ConstraintParticipation(int num_dofs) {
  DRAKE_THROW_UNLESS(num_dofs >= 7);
  DRAKE_THROW_UNLESS(num_dofs % 4 == 0 || num_dofs % 4 == 3);
  const bool has_closed_ends = (num_dofs % 4 == 0);
  num_nodes_ = (num_dofs + 1) / 4;
  num_edges_ = has_closed_ends ? num_nodes_ : num_nodes_ - 1;
}

std::unordered_set<int> ConstraintParticipation::ComputeParticipatingDofs()
    const {
  std::unordered_set<int> participating_dofs;
  for (int node_index : participating_nodes_) {
    participating_dofs.insert(
        {4 * node_index, 4 * node_index + 1, 4 * node_index + 2});
  }
  for (int edge_index : participating_edges_) {
    participating_dofs.insert(4 * edge_index + 3);
  }
  return participating_dofs;
}

multibody::contact_solvers::internal::PartialPermutation
ConstraintParticipation::ComputeDofPermutation() const {
  const int num_dofs = num_nodes_ * 3 + num_edges_;
  return internal::ComputeDofPermutation(num_dofs, ComputeParticipatingDofs());
}

bool ConstraintParticipation::empty() const {
  return participating_nodes_.empty() && participating_edges_.empty();
}

multibody::contact_solvers::internal::PartialPermutation ComputeDofPermutation(
    int num_dofs, const std::unordered_set<int>& participating_dofs) {
  DRAKE_THROW_UNLESS(num_dofs >= 0);
  DRAKE_ASSERT(std::all_of(participating_dofs.begin(), participating_dofs.end(),
                           [=](int dof) {
                             return 0 <= dof && dof < num_dofs;
                           }));
  std::vector<int> permuted_dof_indexes(num_dofs, -1);
  int permuted_dof_index = 0;
  for (int dof = 0; dof < num_dofs; ++dof) {
    if (participating_dofs.contains(dof))
      permuted_dof_indexes[dof] = permuted_dof_index++;
  }
  return multibody::contact_solvers::internal::PartialPermutation(
      std::move(permuted_dof_indexes));
}

}  // namespace internal
}  // namespace der
}  // namespace multibody
}  // namespace drake
