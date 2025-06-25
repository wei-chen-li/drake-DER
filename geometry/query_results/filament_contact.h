#pragma once

#include <memory>
#include <optional>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "drake/common/eigen_types.h"
#include "drake/common/ssize.h"
#include "drake/geometry/geometry_ids.h"
#include "drake/geometry/query_results/contact_surface.h"
#include "drake/math/rotation_matrix.h"

namespace drake {
namespace geometry {
namespace internal {

/* A discrete representation of the intersection of two geometries A and B, at
 least one of which is a filament. We maintain the convention that geometry A is
 always a filament and geometry B may be a filament. When both geometries are
 filaments, we maintain the convention that the GeometryId of geometry A is less
 than or equal to (in the case of self contact) the GeometryId of geometry B.
 @tparam_double_only */
template <typename T>
class FilamentContactGeometryPair {
 public:
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(FilamentContactGeometryPair);

  /* Constructs a filament-filament contact geometry pair with the given data.
   @param[in] id_A
      The GeometryId of the filament geometry A.
   @param[in] id_B
      The GeometryId of the filament geometry B.
   @param[in] p_WCs
      The contact points expressed in the world frame.
   @param[in] nhats_BA_W
      The contact normals pointing from geometry B into geometry A expressed in
      the world frame.
   @param[in] signed_distances
      The signed distances of the contacts.
   @param[in] contact_edge_indexes_A
      The indexes of filament A edges participating in contact, listed in the
      order in accordance with p_WCs and nhats_BA_W.
   @param[in] kinematic_weights_A
      Tuples (𝑤₀, w₁, 𝑤₂) so that the velocity of a point Ac coincident with
      the contact point C and affixed to geometry A, denoted as v_WAc, equals
      𝑤₀ ẋᵢ + w₁ γ̇ⁱ + 𝑤₂ ẋᵢ₊₁.
   @param[in] contact_edge_indexes_B
      The indexes of filament B edges participating in contact, listed in the
      order in accordance with p_WCs and nhats_BA_W.
   @param[in] kinematic_weights_B
      Tuples (𝑤₀, w₁, 𝑤₂) so that the velocity of a point Bc coincident with
      the contact point C and affixed to geometry B, denoted as v_WBc, equals
      𝑤₀ ẋᵢ + w₁ γ̇ⁱ + 𝑤₂ ẋᵢ₊₁.
   @params[in] areas
      (Optional) Areas of the contact patches.
   @params[in] pressures
      (Optional) Pressure at the centroid of the contact patches.
   @params[in] surface_meshes
      (Optional) Surface meshes containing the contact patches.
   @pre `id_A <= id_B`.
   @pre `p_WCs`, `nhats_BA_W`, `signed_distances`, `contact_edge_indexes_A`,
      `kinematic_weights_A`, `contact_edge_indexes_B`, and
      `kinematic_weights_B` have the same size and not empty.
   @pre `areas` and `pressures` have the same size as `p_WCs` if non-null. */
  FilamentContactGeometryPair(
      GeometryId id_A, GeometryId id_B, std::vector<Vector3<T>> p_WCs,
      std::vector<Vector3<T>> nhats_BA_W, std::vector<T> signed_distances,
      std::vector<int> contact_edge_indexes_A,
      std::vector<std::tuple<T, Vector3<T>, T>> kinematic_weights_A,
      std::vector<int> contact_edge_indexes_B,
      std::vector<std::tuple<T, Vector3<T>, T>> kinematic_weights_B,
      std::optional<std::vector<T>> areas = std::nullopt,
      std::optional<std::vector<T>> pressures = std::nullopt,
      std::optional<std::vector<PolygonSurfaceMesh<T>>> surface_meshes =
          std::nullopt);

  /* Constructs a filament-rigid contact geometry pair with the given data.
   @param[in] id_A
      The GeometryId of the filament geometry A.
   @param[in] id_B
      The GeometryId of the rigid geometry B.
   @param[in] p_WCs
      The contact points expressed in the world frame.
   @param[in] nhats_BA_W
      The contact normals pointing from geometry B into geometry A expressed in
      the world frame.
   @param[in] signed_distances
      The signed distances of the contacts.
   @param[in] contact_edge_indexes_A
      The indexes of filament A edges participating in contact, listed in the
      order in accordance with p_WCs and nhats_BA_W.
   @param[in] kinematic_weights_A
      Tuples (𝑤₀, w₁, 𝑤₂) so that the velocity of a point Ac coincident with
      the contact point C and affixed to geometry A, denoted as v_WAc, equals
      𝑤₀ ẋᵢ + w₁ γ̇ⁱ + 𝑤₂ ẋᵢ₊₁.
   @params[in] areas
      (Optional) Areas of the contact patches.
   @params[in] pressures
      (Optional) Pressure at the centroid of the contact patches.
   @params[in] surface_meshes
      (Optional) Surface meshes containing the contact patches.
   @pre `p_WCs`, `nhats_BA_W`, `signed_distances`, `contact_edge_indexes_A`, and
      `kinematic_weights_A` have the same size and not empty.
   @pre `areas` and `pressures` have the same size as `p_WCs` if non-null. */
  FilamentContactGeometryPair(
      GeometryId id_A, GeometryId id_B, std::vector<Vector3<T>> p_WCs,
      std::vector<Vector3<T>> nhats_BA_W, std::vector<T> signed_distances,
      std::vector<int> contact_edge_indexes_A,
      std::vector<std::tuple<T, Vector3<T>, T>> kinematic_weights_A,
      std::optional<std::vector<T>> areas = std::nullopt,
      std::optional<std::vector<T>> pressures = std::nullopt,
      std::optional<std::vector<PolygonSurfaceMesh<T>>> surface_meshes =
          std::nullopt);

  /* Returns the GeometryId of geometry A. If `is_B_filament()` is true, this
   is guaranteed to be less than or equal to id_B(). */
  GeometryId id_A() const { return id_A_; }

  /* Returns the GeometryId of geometry B. If `is_B_filament()` is true, this
   is guaranteed to be greater than or equal to id_A(). */
  GeometryId id_B() const { return id_B_; }

  /* Returns true if geometry B is a filament, false if geometry B is rigid. */
  bool is_B_filament() const { return contact_edge_indexes_B_.has_value(); }

  /* Returns true if the contacts contains patch information. */
  bool is_patch_contact() const { return areas_.has_value(); }

  /* Returns the total number of contact points between this geometry pair. */
  int num_contact_points() const { return ssize(p_WCs_); }

  /* Returns the contact points expressed in the world frame. */
  const std::vector<Vector3<T>>& p_WCs() const { return p_WCs_; }

  /* Returns the contact normals pointing from geometry B into geometry A
   expressed in the world frame. */
  const std::vector<Vector3<T>>& nhats_BA_W() const { return nhats_BA_W_; }

  /* Returns rotation matrices that transform the basis of frame W into the
   basis of an arbitrary frame C. In this transformation, the z-axis of frame,
   Cz, is aligned with the vector n̂. The vector n̂ represents the normal as
   opposite to the one reported in `nhats_BA_W()`. Cx and Cy are arbitrary but
   sufficient to form the right-handed basis. The ordering is same as p_WCs() or
   nhats_BA_W(). */
  const std::vector<math::RotationMatrix<T>>& R_WCs() const { return R_WCs_; }

  /* Returns the signed distances of the contacts. The ordering is same as
   p_WCs() or nhats_BA_W(). */
  const std::vector<T>& signed_distances() const { return signed_distances_; }

  /* Returns the areas of the contact patches. The ordering is same as p_WCs()
   or nhats_BA_W().
   @pre `is_patch_contact()`. */
  const std::vector<T>& areas() const {
    DRAKE_THROW_UNLESS(is_patch_contact());
    return *areas_;
  }

  /* Returns the pressure at the centroid of the contact patches. The ordering
   is same as p_WCs() or nhats_BA_W().
   @pre `is_patch_contact()`. */
  const std::vector<T>& pressures() const {
    DRAKE_THROW_UNLESS(is_patch_contact());
    return *pressures_;
  }

  /* Returns the indexes of filament A edges participating in contact. The
   ordering is same as p_WCs() or nhats_BA_W(). */
  const std::vector<int>& contact_edge_indexes_A() const {
    return contact_edge_indexes_A_;
  }

  /* Returns tuples (𝑤₀, w₁, 𝑤₂) so that the velocity of a point Ac coincident
    with the contact point C and affixed to geometry A, denoted as v_WAc, equals
    𝑤₀ ẋᵢ + w₁ γ̇ⁱ + 𝑤₂ ẋᵢ₊₁ */
  const std::vector<std::tuple<T, Vector3<T>, T>>& kinematic_weights_A() const {
    return kinematic_weights_A_;
  }

  /* Returns the indexes of filament B edges participating in contact. The
   ordering is same as p_WCs() or nhats_BA_W().
   @pre `is_B_filament()`. */
  const std::vector<int>& contact_edge_indexes_B() const {
    DRAKE_THROW_UNLESS(is_B_filament());
    return *contact_edge_indexes_B_;
  }

  /* Returns tuples (𝑤₀, w₁, 𝑤₂) so that the velocity of a point Bc coincident
   with the contact point C and affixed to geometry B, denoted as v_WBc, equals
   𝑤₀ ẋᵢ + w₁ γ̇ⁱ + 𝑤₂ ẋᵢ₊₁.
   @pre `is_B_filament()`. */
  const std::vector<std::tuple<T, Vector3<T>, T>>& kinematic_weights_B() const {
    DRAKE_THROW_UNLESS(is_B_filament());
    return *kinematic_weights_B_;
  }

  /* Returns the surface meshes containing the contact patches.
   @pre `is_patch_contact()`. */
  const std::vector<PolygonSurfaceMesh<T>>& surface_meshes() const {
    DRAKE_THROW_UNLESS(is_patch_contact());
    return *surface_meshes_;
  }

 private:
  GeometryId id_A_;
  GeometryId id_B_;
  std::vector<Vector3<T>> p_WCs_;
  std::vector<Vector3<T>> nhats_BA_W_;
  std::vector<math::RotationMatrix<T>> R_WCs_;
  std::vector<T> signed_distances_;
  std::vector<int> contact_edge_indexes_A_;
  std::vector<std::tuple<T, Vector3<T>, T>> kinematic_weights_A_;
  std::optional<std::vector<int>> contact_edge_indexes_B_;
  std::optional<std::vector<std::tuple<T, Vector3<T>, T>>> kinematic_weights_B_;
  std::optional<std::vector<T>> areas_;
  std::optional<std::vector<T>> pressures_;
  std::optional<std::vector<Vector3<T>>> pressure_gradients_A_W_;
  std::optional<std::vector<Vector3<T>>> pressure_gradients_B_W_;
  std::optional<std::vector<PolygonSurfaceMesh<T>>> surface_meshes_;
};

/* Data structure to hold contact information about filaments.
 @tparam_double_only */
template <typename T>
class FilamentContact {
 public:
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(FilamentContact);

  FilamentContact() = default;

  /* Adds a filament-filament point contact geometry pair to this
   FilamentContact.
   @param[in] id_A
      The GeometryId of the filament geometry A.
   @param[in] id_B
      The GeometryId of the filament geometry B.
   @param[in] p_WCs
      The contact points expressed in the world frame.
   @param[in] nhats_BA_W
      The contact normals pointing from geometry B into geometry A expressed in
      the world frame.
   @param[in] signed_distances
      The signed distances of the contacts.
   @param[in] contact_edge_indexes_A
      The indexes of filament A edges for each contact point.
   @param[in] contact_edge_indexes_B
      The indexes of filament B edges for each contact point.
   @param[in] node_positions_A
      The position of the nodes of filament A. This is required to compute the
      kinematic coordinates.
   @param[in] node_positions_B
      The position of the nodes of filament B. This is required to compute the
      kinematic coordinates.
   @pre `id_A <= id_B`.
   @pre `p_WCs`, `nhats_BA_W`, `signed_distances`, `contact_edge_indexes_A`, and
      `contact_edge_indexes_B` have the same size and not empty. */
  void AddFilamentFilamentContactGeometryPair(
      GeometryId id_A, GeometryId id_B, std::vector<Vector3<T>> p_WCs,
      std::vector<Vector3<T>> nhats_BA_W, std::vector<T> signed_distances,
      std::vector<int> contact_edge_indexes_A,
      std::vector<int> contact_edge_indexes_B,
      const Eigen::Ref<const Eigen::Matrix3X<T>>& node_positions_A,
      const Eigen::Ref<const Eigen::Matrix3X<T>>& node_positions_B);

  /* Adds a filament-rigid point contact geometry pair to this FilamentContact.
   @param[in] id_A
      The GeometryId of the filament geometry A.
   @param[in] id_B
      The GeometryId of the rigid geometry B.
   @param[in] p_WCs
      The contact points expressed in the world frame.
   @param[in] nhats_BA_W
      The contact normals pointing from geometry B into geometry A expressed in
      the world frame.
   @param[in] signed_distances
      The signed distances of the contacts.
   @param[in] contact_edge_indexes_A
      The indexes of filament A edges for each contact point.
   @param[in] node_positions_A
      The position of the nodes of filament A. This is required to compute the
      kinematic coordinates.
   @pre `p_WCs`, `nhats_BA_W`, `signed_distances`, and `contact_edge_indexes_A`
      have the same size and not empty. */
  void AddFilamentRigidContactGeometryPair(
      GeometryId id_A, GeometryId id_B, std::vector<Vector3<T>> p_WCs,
      std::vector<Vector3<T>> nhats_BA_W, std::vector<T> signed_distances,
      std::vector<int> contact_edge_indexes_A,
      const Eigen::Ref<const Eigen::Matrix3X<T>>& node_positions_A);

  /* Adds a filament-filament patch contact geometry pair to this
   FilamentContact.
   @param[in] id_A
      The GeometryId of the filament geometry A.
   @param[in] id_B
      The GeometryId of the filament geometry B.
   @param[in] contact_surfaces
      The contact surfaces between the two geometries.
   @param[in] contact_edge_indexes_A_per_surface
      The indexes of filament A edges for each contact surface.
   @param[in] contact_edge_indexes_B_per_surface
      The indexes of filament B edges for each contact surface.
   @param[in] node_positions_A
      The position of the nodes of filament A. This is required to compute the
      kinematic coordinates.
   @param[in] node_positions_B
      The position of the nodes of filament B. This is required to compute the
      kinematic coordinates.
   @pre `id_A <= id_B`.
   @pre For all `contact_surface` in `contact_surfaces`:
      `contact_surface != nullptr &&
      ((contact_surface->id_M() == id_A && contact_surface->id_N() == id_B) ||
       (contact_surface->id_M() == id_B && contact_surface->id_N() == id_A)) &&
       contact_surface->HasGradE_M() && contact_surface->HasGradE_N() &&
       contact_surface.representation() == kPolygon`. */
  void AddFilamentFilamentContactGeometryPair(
      GeometryId id_A, GeometryId id_B,
      const std::vector<std::unique_ptr<ContactSurface<T>>>& contact_surfaces,
      const std::vector<int>& contact_edge_indexes_A_per_surface,
      const std::vector<int>& contact_edge_indexes_B_per_surface,
      const Eigen::Ref<const Eigen::Matrix3X<T>>& node_positions_A,
      const Eigen::Ref<const Eigen::Matrix3X<T>>& node_positions_B);

  /* Adds a filament-rigid patch contact geometry pair to this FilamentContact.
   @param[in] id_A
      The GeometryId of the filament geometry A.
   @param[in] id_B
      The GeometryId of the filament geometry B.
   @param[in] contact_surfaces
      The contact surfaces between the two geometries.
   @param[in] contact_edge_indexes_A_per_surface
      The indexes of filament A edges for each contact surface.
   @param[in] node_positions_A
      The position of the nodes of filament A. This is required to compute the
      kinematic coordinates.
   @pre For all `contact_surface` in `contact_surfaces`:
      `contact_surface != nullptr &&
      ((contact_surface->id_M() == id_A && contact_surface->id_N() == id_B &&
        contact_surface->HasGradE_M()) ||
       (contact_surface->id_M() == id_B && contact_surface->id_N() == id_A &&
        contact_surface->HasGradE_N())) &&
       contact_surface.representation() == kPolygon`. */
  void AddFilamentRigidContactGeometryPair(
      GeometryId id_A, GeometryId id_B,
      const std::vector<std::unique_ptr<ContactSurface<T>>>& contact_surfaces,
      const std::vector<int>& contact_edge_indexes_A_per_surface,
      const Eigen::Ref<const Eigen::Matrix3X<T>>& node_positions_A);

  /* Returns a list of geometry pairs that are in contact, where at least one of
   the geometries is a filament. Each pair includes associated contact
   information. */
  const std::vector<FilamentContactGeometryPair<T>>& contact_geometry_pairs()
      const {
    return contact_geometry_pairs_;
  }

  /* Returns the set of edge indices for the filament geometry with `id` that
   are under contact. */
  const std::unordered_set<int>& contact_edges(GeometryId id) const;

 private:
  std::unordered_map<GeometryId, std::unordered_set<int>> id_to_contact_edges_;
  std::vector<FilamentContactGeometryPair<T>> contact_geometry_pairs_;
};

}  // namespace internal
}  // namespace geometry
}  // namespace drake
