#pragma once

#include <memory>

#include "drake/common/parallelism.h"
#include "drake/multibody/der/der_state.h"
#include "drake/multibody/der/der_structural_property.h"
#include "drake/multibody/der/der_undeformed_state.h"
#include "drake/multibody/der/energy_hessian_matrix.h"

namespace drake {
namespace multibody {
namespace der {
namespace internal {

/* Computes E = Eₛ + Eₙ + Eₜ.

 @pre `undeformed` and `state` are compatible.
 @tparam_default_scalar */
template <typename T>
T ComputeElasticEnergy(const DerStructuralProperty<T>& prop,
                       const DerUndeformedState<T>& undeformed,
                       const DerState<T>& state);

/* Computes ∂E/∂q.

 @pre `undeformed` and `state` are compatible.
 @pre `jacobian != nullptr`.
 @pre `jacobian->size() == state.num_dofs()`.
 @tparam_default_scalar */
template <typename T>
void ComputeElasticEnergyJacobian(const DerStructuralProperty<T>& prop,
                                  const DerUndeformedState<T>& undeformed,
                                  const DerState<T>& state,
                                  EigenPtr<Eigen::VectorX<T>> jacobian);

/* Computes ∂²E/∂q².

 @pre `undeformed` and `state` are compatible.
 @pre `hessian != nullptr`.
 @pre `hessian` is allocated with the correct number of DoFs.
 @tparam_default_scalar */
template <typename T>
void ComputeElasticEnergyHessian(const DerStructuralProperty<T>& prop,
                                 const DerUndeformedState<T>& undeformed,
                                 const DerState<T>& state,
                                 EnergyHessianMatrix<T>* hessian,
                                 Parallelism parallelism);

/* Computes Eₛ. */
template <typename T>
T ComputeStretchingEnergy(const DerStructuralProperty<T>& prop,
                          const DerUndeformedState<T>& undeformed,
                          const DerState<T>& state);

/* Adds ∂Eₛ/∂q to `jacobian`. */
template <typename T>
void AddStretchingEnergyJacobian(const DerStructuralProperty<T>& prop,
                                 const DerUndeformedState<T>& undeformed,
                                 const DerState<T>& state,
                                 EigenPtr<Eigen::VectorX<T>> jacobian);

/* Adds ∂²Eₛ/∂q² to `hessian`. */
template <typename T>
void AddStretchingEnergyHessian(const DerStructuralProperty<T>& prop,
                                const DerUndeformedState<T>& undeformed,
                                const DerState<T>& state,
                                EnergyHessianMatrix<T>* hessian,
                                Parallelism parallelism);

/* Computes Eₜ. */
template <typename T>
T ComputeTwistingEnergy(const DerStructuralProperty<T>& prop,
                        const DerUndeformedState<T>& undeformed,
                        const DerState<T>& state);

/* Adds ∂Eₜ/∂q to `jacobian`. */
template <typename T>
void AddTwistingEnergyJacobian(const DerStructuralProperty<T>& prop,
                               const DerUndeformedState<T>& undeformed,
                               const DerState<T>& state,
                               EigenPtr<Eigen::VectorX<T>> jacobian);

/* Adds ∂²Eₜ/∂q² to `hessian`. */
template <typename T>
void AddTwistingEnergyHessian(const DerStructuralProperty<T>& prop,
                              const DerUndeformedState<T>& undeformed,
                              const DerState<T>& state,
                              EnergyHessianMatrix<T>* hessian,
                              Parallelism parallelism);

/* Computes Eₙ. */
template <typename T>
T ComputeBendingEnergy(const DerStructuralProperty<T>& prop,
                       const DerUndeformedState<T>& undeformed,
                       const DerState<T>& state);

/* Adds ∂Eₙ/∂q to `jacobian`. */
template <typename T>
void AddBendingEnergyJacobian(const DerStructuralProperty<T>& prop,
                              const DerUndeformedState<T>& undeformed,
                              const DerState<T>& state,
                              EigenPtr<Eigen::VectorX<T>> jacobian);

/* Adds ∂²Eₙ/∂q² to `hessian`. */
template <typename T>
void AddBendingEnergyHessian(const DerStructuralProperty<T>& prop,
                             const DerUndeformedState<T>& undeformed,
                             const DerState<T>& state,
                             EnergyHessianMatrix<T>* hessian,
                             Parallelism parallelism);

}  // namespace internal
}  // namespace der
}  // namespace multibody
}  // namespace drake
