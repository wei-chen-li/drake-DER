#pragma once

// GENERATED FILE DO NOT EDIT
// This file contains docstrings for the Python bindings that were
// automatically extracted by mkdoc.py.

#include <array>
#include <utility>

#if defined(__GNUG__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
#endif

// #include "drake/multibody/der/acceleration_newmark_scheme.h"
// #include "drake/multibody/der/constraint_participation.h"
// #include "drake/multibody/der/damping_model.h"
// #include "drake/multibody/der/der_indexes.h"
// #include "drake/multibody/der/der_model.h"
// #include "drake/multibody/der/der_solver.h"
// #include "drake/multibody/der/der_state.h"
// #include "drake/multibody/der/der_state_system.h"
// #include "drake/multibody/der/der_structural_property.h"
// #include "drake/multibody/der/der_undeformed_state.h"
// #include "drake/multibody/der/dirichlet_boundary_condition.h"
// #include "drake/multibody/der/discrete_time_integrator.h"
// #include "drake/multibody/der/elastic_energy.h"
// #include "drake/multibody/der/energy_hessian_matrix.h"
// #include "drake/multibody/der/external_force_field.h"
// #include "drake/multibody/der/schur_complement.h"
// #include "drake/multibody/der/velocity_newmark_scheme.h"

// Symbol: pydrake_doc_multibody_der
constexpr struct /* pydrake_doc_multibody_der */ {
  // Symbol: drake
  struct /* drake */ {
    // Symbol: drake::multibody
    struct /* multibody */ {
      // Symbol: drake::multibody::der
      struct /* der */ {
        // Symbol: drake::multibody::der::DerEdgeIndex
        struct /* DerEdgeIndex */ {
          // Source: drake/multibody/der/der_indexes.h
          const char* doc = R"""(Type used to index DER edges.)""";
        } DerEdgeIndex;
        // Symbol: drake::multibody::der::DerModel
        struct /* DerModel */ {
          // Source: drake/multibody/der/der_model.h
          const char* doc =
R"""(Discrete Elastic Rod (DER) captures the mechanics of a deformable
filament discretized along its length. A DER is composed of nodes
connected sequentially by edges and is subject to internal elastic
forces arising from stretching, bending, and twisting. A DER can have
either open ends (e.g., to model a rope) or closed ends (e.g., to
model a pulley belt).

For each pair of consecutive nodes xᵢ and xᵢ₊₁, an edge is defined as
eⁱ = xᵢ₊₁ - xᵢ, with an associated tangent vector tⁱ = eⁱ / ‖eⁱ‖.
Subscripts denote quantities associated with nodes, while superscripts
refer to quantities associated with edges. Each edge has an associated
reference frame with directors (d₁ⁱ, d₂ⁱ, tⁱ), and a material frame
with directors (m₁ⁱ, m₂ⁱ, tⁱ). The directors m₁ⁱ and m₂ⁱ align with
the two principal axes of the rod's cross-section, meaning the
material frame rotates with the rod. The angle of rotation from the
reference frame to the material frame is denoted by γⁱ.

A DER with n nodes and open ends has a configuration vector q = [x₀ᵀ
γ⁰ x₁ᵀ ... xₙ₋₁ᵀ]ᵀ with a total of 4n-1 degrees of freedom.

A DER with n nodes and closed ends has a configuration vector q = [x₀ᵀ
γ⁰ x₁ᵀ ... xₙ₋₁ᵀ γⁿ⁻¹]ᵀ with a total of 4n degrees of freedom.

The governing equation for a DER is

M q̈ = Fᵢₙₜ(q, q̇) + Fₑₓₜ,

where M is the lumped mass matrix. The internal force Fᵢₙₜ includes
elastic force, given by −∂E(q)/∂q, where E(q) is the elastic energy,
as well as internal damping forces. The external force Fₑₓₜ includes
effects such as gravity.

``DerModel`` provides a method to evaluate the residual

R(q, q̇, q̈) = M q̈ - Fᵢₙₜ(q, q̇) - Fₑₓₜ.

It also provides a method to compute a weighted sum of the matrices
∂R/∂q, ∂R/∂q̇, and ∂R/∂q̈.)""";
          // Symbol: drake::multibody::der::DerModel::ApplyBoundaryCondition
          struct /* ApplyBoundaryCondition */ {
            // Source: drake/multibody/der/der_model.h
            const char* doc =
R"""(Applies the boundary condition for this DerModel to the ``state``.

Precondition:
    ``state != nullptr``.

Precondition:
    ``state`` is allocated using CreateDerState() of this DerModel.)""";
          } ApplyBoundaryCondition;
          // Symbol: drake::multibody::der::DerModel::Builder
          struct /* Builder */ {
            // Source: drake/multibody/der/der_model.h
            const char* doc =
R"""(DerModel::Builder is a factory class for DerModel. It is single use:
after calling Build(), this builder should be discarded.)""";
            // Symbol: drake::multibody::der::DerModel::Builder::AddEdge
            struct /* AddEdge */ {
              // Source: drake/multibody/der/der_model.h
              const char* doc =
R"""(Connects the current end of the rod to the next node xᵢ₊₁. The
material frame is rotated around the new edge’s tangent by γⁱ relative
to the reference frame. The rod configuration is finalized either when
the new node xᵢ₊₁ coincides with the first node x₀ (resulting in a
closed-ends DER), or when any methods under [Setting the Undeformed
State](set_undeformed_state) is called.

Returns:
    A tuple of indexes of (edge, node).

Note:
    Must be called after AddFirstEdge().)""";
            } AddEdge;
            // Symbol: drake::multibody::der::DerModel::Builder::AddFirstEdge
            struct /* AddFirstEdge */ {
              // Source: drake/multibody/der/der_model.h
              const char* doc =
R"""(Adds the first edge from node x₀ to node x₁. Also specifies the
reference frame director d₁⁰. If d₁⁰ is not specified, it is choosen
as an arbitrary director that is perpendicular to x₁ - x₀. The
material frame director m₁⁰ is the rotation of d₁⁰ around the tangent
by γ⁰.

Returns:
    A tuple of indexes of (node, edge, node).

Precondition:
    d1_0 is a unit vector.

Precondition:
    d1_0 is perpendicular to x_1 - x_0.)""";
            } AddFirstEdge;
            // Symbol: drake::multibody::der::DerModel::Builder::Build
            struct /* Build */ {
              // Source: drake/multibody/der/der_model.h
              const char* doc =
R"""(Builds the DerModel from the configurations and settings sepcified
above.)""";
            } Build;
            // Symbol: drake::multibody::der::DerModel::Builder::Builder
            struct /* ctor */ {
              // Source: drake/multibody/der/der_model.h
              const char* doc = R"""()""";
            } ctor;
            // Symbol: drake::multibody::der::DerModel::Builder::SetCircularCrossSection
            struct /* SetCircularCrossSection */ {
              // Source: drake/multibody/der/der_model.h
              const char* doc = R"""(Sets the cross section to be a circle.)""";
            } SetCircularCrossSection;
            // Symbol: drake::multibody::der::DerModel::Builder::SetDampingCoefficients
            struct /* SetDampingCoefficients */ {
              // Source: drake/multibody/der/der_model.h
              const char* doc =
R"""(Sets the mass coefficient α and the stiffness coefficient β.)""";
            } SetDampingCoefficients;
            // Symbol: drake::multibody::der::DerModel::Builder::SetEllipticalCrossSection
            struct /* SetEllipticalCrossSection */ {
              // Source: drake/multibody/der/der_model.h
              const char* doc =
R"""(Sets the cross section to be a rectangle. One semi-axis with length
``a`` is aligned with the material frame m₁ director; another
semi-axis with length ``b`` is aligned with the material frame m₂
director.)""";
            } SetEllipticalCrossSection;
            // Symbol: drake::multibody::der::DerModel::Builder::SetMaterialProperties
            struct /* SetMaterialProperties */ {
              // Source: drake/multibody/der/der_model.h
              const char* doc =
R"""(Sets the Young's modulus, shear modulus, and mass density.)""";
            } SetMaterialProperties;
            // Symbol: drake::multibody::der::DerModel::Builder::SetRectangularCrossSection
            struct /* SetRectangularCrossSection */ {
              // Source: drake/multibody/der/der_model.h
              const char* doc =
R"""(Sets the cross section to be a rectangle. The ``width`` is aligned
with the material frame m₁ director, and the ``height`` is aligned
with the material frame m₂ director.)""";
            } SetRectangularCrossSection;
            // Symbol: drake::multibody::der::DerModel::Builder::SetUndeformedNaturalCurvatureAndZeroTwist
            struct /* SetUndeformedNaturalCurvatureAndZeroTwist */ {
              // Source: drake/multibody/der/der_model.h
              const char* doc =
R"""(Sets the undeformed edge lengths from the current node positions and
set the undeformed twists to zero. Furthermore, if the DER has open
ends, set the curvature to zero; is the DER has closed ends, set the
curvature to that of a circle.)""";
            } SetUndeformedNaturalCurvatureAndZeroTwist;
            // Symbol: drake::multibody::der::DerModel::Builder::SetUndeformedZeroCurvatureAndZeroTwist
            struct /* SetUndeformedZeroCurvatureAndZeroTwist */ {
              // Source: drake/multibody/der/der_model.h
              const char* doc =
R"""(Sets the undeformed curvatures and twists to zero; the undeformed edge
lengths are computed from the current node positions.)""";
            } SetUndeformedZeroCurvatureAndZeroTwist;
          } Builder;
          // Symbol: drake::multibody::der::DerModel::Clone
          struct /* Clone */ {
            // Source: drake/multibody/der/der_model.h
            const char* doc =
R"""(Creates a deep copy of this DerModel. Even though the cloned model is
functionally identical, any DerState and Scratch created for this
model are not compatible with the cloned model, and vice versa.)""";
          } Clone;
          // Symbol: drake::multibody::der::DerModel::ComputeCenterOfMassPosition
          struct /* ComputeCenterOfMassPosition */ {
            // Source: drake/multibody/der/der_model.h
            const char* doc =
R"""(Computes the position of the center of mass.

Precondition:
    ``state`` is allocated using CreateDerState() of this DerModel.)""";
          } ComputeCenterOfMassPosition;
          // Symbol: drake::multibody::der::DerModel::ComputeCenterOfMassTranslationalVelocity
          struct /* ComputeCenterOfMassTranslationalVelocity */ {
            // Source: drake/multibody/der/der_model.h
            const char* doc =
R"""(Computes the translational velocity of the center of mass.

Precondition:
    ``state`` is allocated using CreateDerState() of this DerModel.)""";
          } ComputeCenterOfMassTranslationalVelocity;
          // Symbol: drake::multibody::der::DerModel::ComputeEffectiveAngularVelocity
          struct /* ComputeEffectiveAngularVelocity */ {
            // Source: drake/multibody/der/der_model.h
            const char* doc =
R"""(Computes the angular velocity about the center of mass.

Precondition:
    ``state`` is allocated using CreateDerState() of this DerModel.)""";
          } ComputeEffectiveAngularVelocity;
          // Symbol: drake::multibody::der::DerModel::ComputeElasticEnergy
          struct /* ComputeElasticEnergy */ {
            // Source: drake/multibody/der/der_model.h
            const char* doc =
R"""(Computes the elastic energy of the DER.

Precondition:
    ``state`` is allocated using CreateDerState() of this DerModel.)""";
          } ComputeElasticEnergy;
          // Symbol: drake::multibody::der::DerModel::ComputeResidual
          struct /* ComputeResidual */ {
            // Source: drake/multibody/der/der_model.h
            const char* doc =
R"""(Computes the residual R(q, q̇, q̈) (see class doc) evaluated at the
given ``state`` and under the ``external_force_field``.

The residual for degrees of freedom under the Dirichlet boundary
condition is set to zero. Therefore their residual should not be used
as a metric for the error on the boundary condition.

Parameter ``state``:
    The ``DerState`` at which the residual is evaluated.

Parameter ``external_force_field``:
    The external force field to evaluate under.

Parameter ``scratch``:
    The scratch allocated using MakeScratch().

Returns:
    The residual vector.

Precondition:
    ``state`` is allocated using CreateDerState() of this DerModel.

Precondition:
    ``scratch != nullptr``.

Precondition:
    ``scratch`` is allocated using MakeScratch() of this DerModel.)""";
          } ComputeResidual;
          // Symbol: drake::multibody::der::DerModel::ComputeTangentMatrix
          struct /* ComputeTangentMatrix */ {
            // Source: drake/multibody/der/der_model.h
            const char* doc =
R"""(Computes an approximated tangent matrix evaluated at the given
``state``.

The tangent matrix is given by a weighted sum of the stiffness matrix
(∂R/∂q), damping matrix (∂R/∂q̇), and mass matrix (∂R/∂q̈).

The rows and columns in the tangent matrix corresponding to the
degrees of freedom under the Dirichlet boundary condition is set to
zero with the exception of the diagonal entries which is set to one.

Parameter ``state``:
    The ``DerState`` at which the tangent matrix is evaluated.

Parameter ``weights``:
    The weights used to combine stiffness, damping, and mass matrices
    (in that order) into the tangent matrix.

Parameter ``scratch``:
    The scratch allocated using MakeScratch().

Returns:
    ``result`` The tangent matrix represented by a
    Block4x4SparseSymmetricMatrix.

Precondition:
    ``state`` is allocated using CreateDerState() of this DerModel.

Precondition:
    ``scratch != nullptr``.

Precondition:
    ``scratch`` is allocated using MakeScratch() of this DerModel.)""";
          } ComputeTangentMatrix;
          // Symbol: drake::multibody::der::DerModel::CreateDerState
          struct /* CreateDerState */ {
            // Source: drake/multibody/der/der_model.h
            const char* doc =
R"""(Creates a default DerState compatible with this DER model.)""";
          } CreateDerState;
          // Symbol: drake::multibody::der::DerModel::DerModel<T>
          struct /* ctor */ {
            // Source: drake/multibody/der/der_model.h
            const char* doc = R"""()""";
          } ctor;
          // Symbol: drake::multibody::der::DerModel::FixPosition
          struct /* FixPosition */ {
            // Source: drake/multibody/der/der_model.h
            const char* doc =
R"""(Fixes the node position or the edge angle indexed by ``index``.

Precondition:
    ``index`` is within the node index or edge index range.)""";
          } FixPosition;
          // Symbol: drake::multibody::der::DerModel::IsPositionFixed
          struct /* IsPositionFixed */ {
            // Source: drake/multibody/der/der_model.h
            const char* doc = R"""()""";
          } IsPositionFixed;
          // Symbol: drake::multibody::der::DerModel::MakeScratch
          struct /* MakeScratch */ {
            // Source: drake/multibody/der/der_model.h
            const char* doc =
R"""(Creates a stratch that can be used by ComputeResidual() and
ComputeTangentMatrix().)""";
          } MakeScratch;
          // Symbol: drake::multibody::der::DerModel::ScratchDeleter
          struct /* ScratchDeleter */ {
            // Source: drake/multibody/der/der_model.h
            const char* doc = R"""()""";
            // Symbol: drake::multibody::der::DerModel::ScratchDeleter::operator()
            struct /* operator_call */ {
              // Source: drake/multibody/der/der_model.h
              const char* doc = R"""()""";
            } operator_call;
          } ScratchDeleter;
          // Symbol: drake::multibody::der::DerModel::ToScalarType
          struct /* ToScalarType */ {
            // Source: drake/multibody/der/der_model.h
            const char* doc =
R"""(Creates a deep copy of this DerModel, transmogrified to use the scalar
type selected by a template parameter.

Raises:
    RuntimeError if this model does not support the destination type.

Template parameter ``U``:
    The destination scalar type, which must be one of the default
    nonsymbolic scalars.)""";
          } ToScalarType;
          // Symbol: drake::multibody::der::DerModel::ValidateDerState
          struct /* ValidateDerState */ {
            // Source: drake/multibody/der/der_model.h
            const char* doc =
R"""((Internal use only) Checks whether the given ``state`` is created from
``this`` DerModel.)""";
          } ValidateDerState;
          // Symbol: drake::multibody::der::DerModel::has_closed_ends
          struct /* has_closed_ends */ {
            // Source: drake/multibody/der/der_model.h
            const char* doc =
R"""(Returns true if this DER has closed ends.)""";
          } has_closed_ends;
          // Symbol: drake::multibody::der::DerModel::mutable_structural_property
          struct /* mutable_structural_property */ {
            // Source: drake/multibody/der/der_model.h
            const char* doc =
R"""(Returns the mutable structural property of this model.)""";
          } mutable_structural_property;
          // Symbol: drake::multibody::der::DerModel::mutable_undeformed_state
          struct /* mutable_undeformed_state */ {
            // Source: drake/multibody/der/der_model.h
            const char* doc =
R"""(Returns the mutable undeformed state of this model.)""";
          } mutable_undeformed_state;
          // Symbol: drake::multibody::der::DerModel::num_dofs
          struct /* num_dofs */ {
            // Source: drake/multibody/der/der_model.h
            const char* doc =
R"""(Returns the number of degrees of freedom in this DER.)""";
          } num_dofs;
          // Symbol: drake::multibody::der::DerModel::num_edges
          struct /* num_edges */ {
            // Source: drake/multibody/der/der_model.h
            const char* doc =
R"""(Returns the number of edges in this DER.)""";
          } num_edges;
          // Symbol: drake::multibody::der::DerModel::num_nodes
          struct /* num_nodes */ {
            // Source: drake/multibody/der/der_model.h
            const char* doc =
R"""(Returns the number of nodes in this DER.)""";
          } num_nodes;
          // Symbol: drake::multibody::der::DerModel::parallelism
          struct /* parallelism */ {
            // Source: drake/multibody/der/der_model.h
            const char* doc =
R"""((Internal use only) Returns the parallelism that ``this`` DerModel
uses when opportunities for parallel computation arises.)""";
          } parallelism;
          // Symbol: drake::multibody::der::DerModel::set_parallelism
          struct /* set_parallelism */ {
            // Source: drake/multibody/der/der_model.h
            const char* doc =
R"""((Internal use only) Configures the parallelism that ``this`` DerModel
uses when opportunities for parallel computation arises.)""";
          } set_parallelism;
          // Symbol: drake::multibody::der::DerModel::structural_property
          struct /* structural_property */ {
            // Source: drake/multibody/der/der_model.h
            const char* doc =
R"""(Returns the structural property of this model.)""";
          } structural_property;
        } DerModel;
        // Symbol: drake::multibody::der::DerNodeIndex
        struct /* DerNodeIndex */ {
          // Source: drake/multibody/der/der_indexes.h
          const char* doc = R"""(Type used to index DER nodes.)""";
        } DerNodeIndex;
        // Symbol: drake::multibody::der::DerStructuralProperty
        struct /* DerStructuralProperty */ {
          // Source: drake/multibody/der/der_structural_property.h
          const char* doc =
R"""(``DerStructuralProperty`` holds properties regarding the Young's
modulus, shear modulus, mass density, and cross section of a discrete
elastic rod.)""";
          // Symbol: drake::multibody::der::DerStructuralProperty::A
          struct /* A */ {
            // Source: drake/multibody/der/der_structural_property.h
            const char* doc = R"""(∫dA.)""";
          } A;
          // Symbol: drake::multibody::der::DerStructuralProperty::DerStructuralProperty<T>
          struct /* ctor */ {
            // Source: drake/multibody/der/der_structural_property.h
            const char* doc = R"""()""";
          } ctor;
          // Symbol: drake::multibody::der::DerStructuralProperty::EA
          struct /* EA */ {
            // Source: drake/multibody/der/der_structural_property.h
            const char* doc = R"""(Young's modulus times ∫dA.)""";
          } EA;
          // Symbol: drake::multibody::der::DerStructuralProperty::EI1
          struct /* EI1 */ {
            // Source: drake/multibody/der/der_structural_property.h
            const char* doc = R"""(Young's modulus times ∫(p⋅m₁)²dA.)""";
          } EI1;
          // Symbol: drake::multibody::der::DerStructuralProperty::EI2
          struct /* EI2 */ {
            // Source: drake/multibody/der/der_structural_property.h
            const char* doc = R"""(Young's modulus times ∫(p⋅m₂)²dA.)""";
          } EI2;
          // Symbol: drake::multibody::der::DerStructuralProperty::FromCircularCrossSection
          struct /* FromCircularCrossSection */ {
            // Source: drake/multibody/der/der_structural_property.h
            const char* doc =
R"""(Create a ``DerStructuralProperty`` corresponding to a circular cross
section with radius ``r``.)""";
          } FromCircularCrossSection;
          // Symbol: drake::multibody::der::DerStructuralProperty::FromEllipticalCrossSection
          struct /* FromEllipticalCrossSection */ {
            // Source: drake/multibody/der/der_structural_property.h
            const char* doc =
R"""(Create a ``DerStructuralProperty`` corresponding to an elliptical
cross section shown in the following figure.


.. raw:: html

    <details><summary>Click to expand C++ code...</summary>

.. code-block:: c++

    m₂
    ↑
    +----+----+
    /     b     \
    |      +--a---+-→ m₁
    \           /
    +---------+

.. raw:: html

    </details>

m₁ and m₂ are the orthonormal material frame directors perpendicular
to the tangent.)""";
          } FromEllipticalCrossSection;
          // Symbol: drake::multibody::der::DerStructuralProperty::FromRectangularCrossSection
          struct /* FromRectangularCrossSection */ {
            // Source: drake/multibody/der/der_structural_property.h
            const char* doc =
R"""(Create a ``DerStructuralProperty`` corresponding to a rectangular
cross section shown in the following figure.


.. raw:: html

    <details><summary>Click to expand C++ code...</summary>

.. code-block:: c++

    m₂
    ↑
    +------+------+
    |      |      |
    height |      +------+-→ m₁
    |             |
    +-------------+
    width

.. raw:: html

    </details>

m₁ and m₂ are the orthonormal material frame directors perpendicular
to the tangent.)""";
          } FromRectangularCrossSection;
          // Symbol: drake::multibody::der::DerStructuralProperty::GJ
          struct /* GJ */ {
            // Source: drake/multibody/der/der_structural_property.h
            const char* doc =
R"""(Shear modulus times ∫((p⋅m₁)²+(p⋅m₂)²)dA.)""";
          } GJ;
          // Symbol: drake::multibody::der::DerStructuralProperty::I1
          struct /* I1 */ {
            // Source: drake/multibody/der/der_structural_property.h
            const char* doc = R"""(∫(p⋅m₁)²dA.)""";
          } I1;
          // Symbol: drake::multibody::der::DerStructuralProperty::I2
          struct /* I2 */ {
            // Source: drake/multibody/der/der_structural_property.h
            const char* doc = R"""(∫(p⋅m₂)²dA.)""";
          } I2;
          // Symbol: drake::multibody::der::DerStructuralProperty::ToScalarType
          struct /* ToScalarType */ {
            // Source: drake/multibody/der/der_structural_property.h
            const char* doc = R"""()""";
          } ToScalarType;
          // Symbol: drake::multibody::der::DerStructuralProperty::rhoA
          struct /* rhoA */ {
            // Source: drake/multibody/der/der_structural_property.h
            const char* doc = R"""(Mass density times ∫dA.)""";
          } rhoA;
          // Symbol: drake::multibody::der::DerStructuralProperty::rhoJ
          struct /* rhoJ */ {
            // Source: drake/multibody/der/der_structural_property.h
            const char* doc =
R"""(Mass density times ∫√((p⋅m₁)²+(p⋅m₂)²)dA.)""";
          } rhoJ;
          // Symbol: drake::multibody::der::DerStructuralProperty::set_A
          struct /* set_A */ {
            // Source: drake/multibody/der/der_structural_property.h
            const char* doc = R"""()""";
          } set_A;
          // Symbol: drake::multibody::der::DerStructuralProperty::set_I1
          struct /* set_I1 */ {
            // Source: drake/multibody/der/der_structural_property.h
            const char* doc = R"""()""";
          } set_I1;
          // Symbol: drake::multibody::der::DerStructuralProperty::set_I2
          struct /* set_I2 */ {
            // Source: drake/multibody/der/der_structural_property.h
            const char* doc = R"""()""";
          } set_I2;
        } DerStructuralProperty;
        // Symbol: drake::multibody::der::DerUndeformedState
        struct /* DerUndeformedState */ {
          // Source: drake/multibody/der/der_undeformed_state.h
          const char* doc =
R"""(``DerUndeformedState`` describes the undeformed state of a discrete
elsatic rod, that is the state without external forces and under
static equilibrium. For example, the undeformed state of a rope may be
in a straight line (zero curvature and zero twist); or the undeformed
state of a rubber band in a circular shape (constant curvature and
zero twist).)""";
          // Symbol: drake::multibody::der::DerUndeformedState::DerUndeformedState<T>
          struct /* ctor */ {
            // Source: drake/multibody/der/der_undeformed_state.h
            const char* doc = R"""()""";
          } ctor;
          // Symbol: drake::multibody::der::DerUndeformedState::NaturalCurvatureZeroTwist
          struct /* NaturalCurvatureZeroTwist */ {
            // Source: drake/multibody/der/der_undeformed_state.h
            const char* doc =
R"""(Creates an undeformed state from the `state`'s current edge length and
zero twist. Furthermore, the curvature is set to zero if the DER has
open ends; set to that of a circle if the DER has closed ends.)""";
          } NaturalCurvatureZeroTwist;
          // Symbol: drake::multibody::der::DerUndeformedState::ToScalarType
          struct /* ToScalarType */ {
            // Source: drake/multibody/der/der_undeformed_state.h
            const char* doc = R"""()""";
          } ToScalarType;
          // Symbol: drake::multibody::der::DerUndeformedState::ZeroCurvatureAndTwist
          struct /* ZeroCurvatureAndTwist */ {
            // Source: drake/multibody/der/der_undeformed_state.h
            const char* doc =
R"""(Creates an undeformed state with the specified edge lengths, zero
curvature, and zero twist.

Precondition:
    Each entry in ``edge_length`` is greater than 0.)""";
          } ZeroCurvatureAndTwist;
          // Symbol: drake::multibody::der::DerUndeformedState::get_curvature_kappa1
          struct /* get_curvature_kappa1 */ {
            // Source: drake/multibody/der/der_undeformed_state.h
            const char* doc = R"""()""";
          } get_curvature_kappa1;
          // Symbol: drake::multibody::der::DerUndeformedState::get_curvature_kappa2
          struct /* get_curvature_kappa2 */ {
            // Source: drake/multibody/der/der_undeformed_state.h
            const char* doc = R"""()""";
          } get_curvature_kappa2;
          // Symbol: drake::multibody::der::DerUndeformedState::get_edge_length
          struct /* get_edge_length */ {
            // Source: drake/multibody/der/der_undeformed_state.h
            const char* doc = R"""(@name Property of edges when undeformed)""";
          } get_edge_length;
          // Symbol: drake::multibody::der::DerUndeformedState::get_twist
          struct /* get_twist */ {
            // Source: drake/multibody/der/der_undeformed_state.h
            const char* doc = R"""()""";
          } get_twist;
          // Symbol: drake::multibody::der::DerUndeformedState::get_voronoi_length
          struct /* get_voronoi_length */ {
            // Source: drake/multibody/der/der_undeformed_state.h
            const char* doc =
R"""(@name Properties of internal nodes when undeformed)""";
          } get_voronoi_length;
          // Symbol: drake::multibody::der::DerUndeformedState::has_closed_ends
          struct /* has_closed_ends */ {
            // Source: drake/multibody/der/der_undeformed_state.h
            const char* doc = R"""()""";
          } has_closed_ends;
          // Symbol: drake::multibody::der::DerUndeformedState::num_dofs
          struct /* num_dofs */ {
            // Source: drake/multibody/der/der_undeformed_state.h
            const char* doc = R"""()""";
          } num_dofs;
          // Symbol: drake::multibody::der::DerUndeformedState::num_edges
          struct /* num_edges */ {
            // Source: drake/multibody/der/der_undeformed_state.h
            const char* doc = R"""()""";
          } num_edges;
          // Symbol: drake::multibody::der::DerUndeformedState::num_internal_nodes
          struct /* num_internal_nodes */ {
            // Source: drake/multibody/der/der_undeformed_state.h
            const char* doc = R"""()""";
          } num_internal_nodes;
          // Symbol: drake::multibody::der::DerUndeformedState::num_nodes
          struct /* num_nodes */ {
            // Source: drake/multibody/der/der_undeformed_state.h
            const char* doc = R"""()""";
          } num_nodes;
          // Symbol: drake::multibody::der::DerUndeformedState::set_curvature_angle
          struct /* set_curvature_angle */ {
            // Source: drake/multibody/der/der_undeformed_state.h
            const char* doc =
R"""(Sets the undeformed curvature components from the angle values.
Equivalent to ``set_curvature_kappa(2*tan(angle1/2),
2*tan(angle2/2))``

Precondition:
    ``angle1.size() == num_internal_nodes()``.

Precondition:
    ``angle2.size() == num_internal_nodes()``.

Precondition:
    Entries in ``angle1`` and ``angle2`` are all within the range
    (-2π, 2π).)""";
          } set_curvature_angle;
          // Symbol: drake::multibody::der::DerUndeformedState::set_curvature_kappa
          struct /* set_curvature_kappa */ {
            // Source: drake/multibody/der/der_undeformed_state.h
            const char* doc =
R"""(Sets the undeformed curvature components from the kappa values.

Precondition:
    ``kappa1.size() == num_internal_nodes()``.

Precondition:
    ``kappa2.size() == num_internal_nodes()``.)""";
          } set_curvature_kappa;
          // Symbol: drake::multibody::der::DerUndeformedState::set_edge_length
          struct /* set_edge_length */ {
            // Source: drake/multibody/der/der_undeformed_state.h
            const char* doc =
R"""(Sets the undeformed edge length.

Precondition:
    ``edge_length.size() == num_edges()``.

Precondition:
    Each entry in ``edge_length`` is greater than 0.)""";
          } set_edge_length;
          // Symbol: drake::multibody::der::DerUndeformedState::set_twist
          struct /* set_twist */ {
            // Source: drake/multibody/der/der_undeformed_state.h
            const char* doc =
R"""(Sets the undeformed twist.

Precondition:
    ``twist.size() == num_internal_nodes()``.)""";
          } set_twist;
        } DerUndeformedState;
      } der;
    } multibody;
  } drake;
} pydrake_doc_multibody_der;

#if defined(__GNUG__)
#pragma GCC diagnostic pop
#endif
