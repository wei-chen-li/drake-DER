#pragma once

#include <initializer_list>
#include <unordered_set>

#include "drake/multibody/contact_solvers/sap/partial_permutation.h"

namespace drake {
namespace multibody {
namespace der {
namespace internal {

class ConstraintParticipation {
 public:
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(ConstraintParticipation);

  /* Constructs a ConstraintParticipation for a discrete elastic rod based on
   the `num_dofs` of the DER.
  @pre `num_dofs >= 7`.
  @pre `num_dofs % 4 == 0 || num_dofs % 4 == 3`.*/
  explicit ConstraintParticipation(int num_dofs);

  /* Mark the given nodes as participating in constraint.
   @pre Each entry in `nodes` is non-negative and less than `num_nodes` provided
   at construction. */
  template <typename Container = std::initializer_list<int>>
  void ParticipateNodes(const Container& nodes) {
    static_assert(std::is_same_v<typename Container::value_type, int>);
    for (int i : nodes) {
      DRAKE_THROW_UNLESS(0 <= i && i < num_nodes_);
      participating_nodes_.insert(i);
    }
  }

  /* Mark the given edges as participating in constraint.
   @pre Each entry in `edges` is non-negative and less than `num_edges` provided
   at construction. */
  template <typename Container = std::initializer_list<int>>
  void ParticipateEdges(const Container& edges) {
    static_assert(std::is_same_v<typename Container::value_type, int>);
    for (int i : edges) {
      DRAKE_THROW_UNLESS(0 <= i && i < num_edges_);
      participating_edges_.insert(i);
    }
  }

  /* Mark the given edges and the nodes adjacent to the edges as participating
   in constraint.
   @pre Each entry in `edges` is non-negative and less than `num_edges` provided
   at construction. */
  template <typename Container = std::initializer_list<int>>
  void ParticipateEdgesAndAdjacentNodes(const Container& edges) {
    static_assert(std::is_same_v<typename Container::value_type, int>);
    for (int i : edges) {
      DRAKE_THROW_UNLESS(0 <= i && i < num_edges_);
      participating_edges_.insert(i);
      /* The two adjacent nodes also participate in constraint. */
      participating_nodes_.insert(i);
      participating_nodes_.insert((i + 1) % num_nodes_);
    }
  }

  /* Returns the set of participating DoFs. */
  std::unordered_set<int> ComputeParticipatingDofs() const;

  /* Returns the DoF partial permutation. The DoF partial permutation p is
   such that p(i) gives the permuted DoF index for DoF i, if DoF i is
   participating in constraint. If both DoF i and DoF j are participating in
   contact, and i < j, then p(i) < p(j). */
  multibody::contact_solvers::internal::PartialPermutation
  ComputeDofPermutation() const;

  /* Returns true if there are no nodes or edges under constraint. */
  [[nodiscard]] bool empty() const;

 private:
  int num_nodes_;
  int num_edges_;
  std::unordered_set<int> participating_nodes_;
  std::unordered_set<int> participating_edges_;
};

/* Returns the DoF partial permutation. The DoF partial permutation p is
 such that p(i) gives the permuted DoF index for DoF i, if DoF i is
 in `participating_dofs`. If both DoF i and DoF j are in `participating_dofs`,
 and i < j, then p(i) < p(j).
 @pre Every dof index in `participating_dofs` is greater than or equal to zero
      and less than `num_dofs`. */
multibody::contact_solvers::internal::PartialPermutation ComputeDofPermutation(
    int num_dofs, const std::unordered_set<int>& participating_dofs);

}  // namespace internal
}  // namespace der
}  // namespace multibody
}  // namespace drake
