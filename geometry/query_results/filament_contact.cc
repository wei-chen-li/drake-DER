#include "drake/geometry/query_results/filament_contact.h"

#include <algorithm>
#include <limits>

namespace drake {
namespace geometry {
namespace internal {

namespace {

/* Compute the tuples (𝑤₀, w₁, 𝑤₂) so that the velocity of the point Gc
 coincident with the contact point C and affixed to geometry G, denoted as
 v_WGc, equals 𝑤₀ ẋᵢ + w₁ γ̇ⁱ + 𝑤₂ ẋᵢ₊₁. */
template <typename T>
std::vector<std::tuple<T, Vector3<T>, T>> ComputeKinematicWeights(
    const std::vector<Vector3<T>>& p_WCs,
    const std::vector<int>& contact_edge_indexes,
    const Eigen::Ref<const Eigen::Matrix3X<T>>& node_positions) {
  DRAKE_THROW_UNLESS(p_WCs.size() == contact_edge_indexes.size());
  const int num_nodes = node_positions.cols();
  std::vector<std::tuple<T, Vector3<T>, T>> kinematic_weights;
  kinematic_weights.reserve(p_WCs.size());
  for (int k = 0; k < ssize(p_WCs); ++k) {
    const int i = contact_edge_indexes[k];
    const int ip1 = (i + 1) % num_nodes;
    const Vector3<T> node_i = node_positions.col(i);
    const Vector3<T> node_ip1 = node_positions.col(ip1);
    const T l = (node_ip1 - node_i).norm();
    const Vector3<T> t = (node_ip1 - node_i) / l;
    const T w2 = (p_WCs[k] - node_i).dot(t) / l;
    const T w0 = 1.0 - w2;
    const Vector3<T> w1 = t.cross(p_WCs[k] - node_i);
    kinematic_weights.emplace_back(w0, w1, w2);
  }
  return kinematic_weights;
}

/* Computes the pressure and signed distance for the `face` in `s`. */
template <typename T>
std::tuple<T, T> ComputePressureAndSignedDistance(const ContactSurface<T>& s,
                                                  int face) {
  const Vector3<T>& p_WC = s.centroid(face);
  const Vector3<T> tri_centroid_barycentric(1 / 3., 1 / 3., 1 / 3.);
  const T p0 = s.is_triangle()
                   ? s.tri_e_MN().Evaluate(face, tri_centroid_barycentric)
                   : s.poly_e_MN().EvaluateCartesian(face, p_WC);

  const Vector3<T>& nhat_NM_W = s.face_normal(face);
  /* The pressure gradients are positive in the direction into the bodies. */
  const T gM = s.HasGradE_M() ? s.EvaluateGradE_M_W(face).dot(nhat_NM_W)
                              : T(std::numeric_limits<double>::infinity());
  const T gN = s.HasGradE_N() ? s.EvaluateGradE_N_W(face).dot(-nhat_NM_W)
                              : T(std::numeric_limits<double>::infinity());
  /* Effective hydroelastic pressure gradient g=gN*gM/(gN+gM) as a result of
   compliant-compliant interaction, see [Masterjohn 2022].
   [Masterjohn 2022] Velocity Level Approximation of Pressure Field Contact
   Patches. */
  const T g = 1.0 / (1.0 / gM + 1.0 / gN);
  /* phi < 0 when in penetration. */
  const T phi0 = -p0 / g;

  return {p0, phi0};
}

/* Removes the elements in `vector` corresponding to the indices in
 `remove_indices`. */
template <typename T>
void RemoveElements(std::vector<T>* vector,
                    const std::vector<int>& remove_indices) {
  DRAKE_THROW_UNLESS(vector != nullptr);
  if (remove_indices.empty()) return;
  DRAKE_ASSERT(std::is_sorted(remove_indices.begin(), remove_indices.end()));
  for (auto it = remove_indices.rbegin(); it != remove_indices.rend(); ++it) {
    const int remove_index = *it;
    vector->erase(vector->begin() + remove_index);
  }
}

}  // namespace

template <typename T>
FilamentContactGeometryPair<T>::FilamentContactGeometryPair(
    GeometryId id_A, GeometryId id_B, std::vector<Vector3<T>> p_WCs,
    std::vector<Vector3<T>> nhats_BA_W, std::vector<T> signed_distances,
    std::vector<int> contact_edge_indexes_A,
    std::vector<std::tuple<T, Vector3<T>, T>> kinematic_weights_A,
    std::vector<int> contact_edge_indexes_B,
    std::vector<std::tuple<T, Vector3<T>, T>> kinematic_weights_B,
    std::optional<std::vector<T>> areas,
    std::optional<std::vector<T>> pressures,
    std::optional<std::vector<PolygonSurfaceMesh<T>>> surface_meshes)
    : id_A_(id_A),
      id_B_(id_B),
      p_WCs_(std::move(p_WCs)),
      nhats_BA_W_(std::move(nhats_BA_W)),
      signed_distances_(std::move(signed_distances)),
      contact_edge_indexes_A_(std::move(contact_edge_indexes_A)),
      kinematic_weights_A_(std::move(kinematic_weights_A)),
      contact_edge_indexes_B_(std::move(contact_edge_indexes_B)),
      kinematic_weights_B_(std::move(kinematic_weights_B)),
      areas_(std::move(areas)),
      pressures_(std::move(pressures)),
      surface_meshes_(std::move(surface_meshes)) {
  DRAKE_THROW_UNLESS(id_A.get_value() <= id_B.get_value());
  DRAKE_THROW_UNLESS(num_contact_points() > 0);
  DRAKE_THROW_UNLESS(ssize(nhats_BA_W_) == num_contact_points());
  DRAKE_THROW_UNLESS(ssize(signed_distances_) == num_contact_points());
  DRAKE_THROW_UNLESS(ssize(contact_edge_indexes_A_) == num_contact_points());
  DRAKE_THROW_UNLESS(ssize(kinematic_weights_A_) == num_contact_points());
  DRAKE_THROW_UNLESS(ssize(*contact_edge_indexes_B_) == num_contact_points());
  DRAKE_THROW_UNLESS(ssize(*kinematic_weights_B_) == num_contact_points());
  if (is_patch_contact()) {
    DRAKE_THROW_UNLESS(areas_.has_value() &&
                       ssize(*areas_) == num_contact_points());
    DRAKE_THROW_UNLESS(pressures_.has_value() &&
                       ssize(*pressures_) == num_contact_points());
    DRAKE_THROW_UNLESS(surface_meshes_.has_value());
  }

  std::vector<int> remove_indices;
  R_WCs_.reserve(num_contact_points());
  for (int i = 0; i < num_contact_points(); ++i) {
    constexpr int kZAxis = 2;
    if (!nhats_BA_W_[i].isZero()) {
      R_WCs_.emplace_back(math::RotationMatrix<T>::MakeFromOneUnitVector(
          -nhats_BA_W_[i], kZAxis));
    } else {
      /* If two geometries are touching exactly on the surface,
       flexible-collision-library may register a collision but give a zero
       normal vector. We remove such contacts. */
      R_WCs_.emplace_back(math::RotationMatrix<T>());
      remove_indices.push_back(i);
    }
  }
  RemoveElements(&p_WCs_, remove_indices);
  RemoveElements(&nhats_BA_W_, remove_indices);
  RemoveElements(&signed_distances_, remove_indices);
  RemoveElements(&contact_edge_indexes_A_, remove_indices);
  RemoveElements(&kinematic_weights_A_, remove_indices);
  RemoveElements(&contact_edge_indexes_B_.value(), remove_indices);
  RemoveElements(&kinematic_weights_B_.value(), remove_indices);
  RemoveElements(&R_WCs_, remove_indices);
  if (is_patch_contact()) {
    RemoveElements(&areas_.value(), remove_indices);
    RemoveElements(&pressures_.value(), remove_indices);
  }
}

template <typename T>
FilamentContactGeometryPair<T>::FilamentContactGeometryPair(
    GeometryId id_A, GeometryId id_B, std::vector<Vector3<T>> p_WCs,
    std::vector<Vector3<T>> nhats_BA_W, std::vector<T> signed_distances,
    std::vector<int> contact_edge_indexes_A,
    std::vector<std::tuple<T, Vector3<T>, T>> kinematic_weights_A,
    std::optional<std::vector<T>> areas,
    std::optional<std::vector<T>> pressures,
    std::optional<std::vector<PolygonSurfaceMesh<T>>> surface_meshes)
    : id_A_(id_A),
      id_B_(id_B),
      p_WCs_(std::move(p_WCs)),
      nhats_BA_W_(std::move(nhats_BA_W)),
      signed_distances_(std::move(signed_distances)),
      contact_edge_indexes_A_(std::move(contact_edge_indexes_A)),
      kinematic_weights_A_(std::move(kinematic_weights_A)),
      areas_(std::move(areas)),
      pressures_(std::move(pressures)),
      surface_meshes_(std::move(surface_meshes)) {
  DRAKE_THROW_UNLESS(num_contact_points() > 0);
  DRAKE_THROW_UNLESS(ssize(nhats_BA_W_) == num_contact_points());
  DRAKE_THROW_UNLESS(ssize(signed_distances_) == num_contact_points());
  DRAKE_THROW_UNLESS(ssize(contact_edge_indexes_A_) == num_contact_points());
  DRAKE_THROW_UNLESS(ssize(kinematic_weights_A_) == num_contact_points());
  if (is_patch_contact()) {
    DRAKE_THROW_UNLESS(areas_.has_value() &&
                       ssize(*areas_) == num_contact_points());
    DRAKE_THROW_UNLESS(pressures_.has_value() &&
                       ssize(*pressures_) == num_contact_points());
    DRAKE_THROW_UNLESS(surface_meshes_.has_value());
  }

  std::vector<int> remove_indices;
  R_WCs_.reserve(num_contact_points());
  for (int i = 0; i < num_contact_points(); ++i) {
    constexpr int kZAxis = 2;
    if (!nhats_BA_W_[i].isZero()) {
      R_WCs_.emplace_back(math::RotationMatrix<T>::MakeFromOneUnitVector(
          -nhats_BA_W_[i], kZAxis));
    } else {
      /* If two geometries are touching exactly on the surface,
       flexible-collision-library may register a collision but give a zero
       normal vector. We remove such contacts. */
      R_WCs_.emplace_back(math::RotationMatrix<T>());
      remove_indices.push_back(i);
    }
  }
  RemoveElements(&p_WCs_, remove_indices);
  RemoveElements(&nhats_BA_W_, remove_indices);
  RemoveElements(&signed_distances_, remove_indices);
  RemoveElements(&contact_edge_indexes_A_, remove_indices);
  RemoveElements(&kinematic_weights_A_, remove_indices);
  RemoveElements(&R_WCs_, remove_indices);
  if (is_patch_contact()) {
    RemoveElements(&areas_.value(), remove_indices);
    RemoveElements(&pressures_.value(), remove_indices);
  }
}

template <typename T>
void FilamentContact<T>::AddFilamentFilamentContactGeometryPair(
    GeometryId id_A, GeometryId id_B, std::vector<Vector3<T>> p_WCs,
    std::vector<Vector3<T>> nhats_BA_W, std::vector<T> signed_distances,
    std::vector<int> contact_edge_indexes_A,
    std::vector<int> contact_edge_indexes_B,
    const Eigen::Ref<const Eigen::Matrix3X<T>>& node_positions_A,
    const Eigen::Ref<const Eigen::Matrix3X<T>>& node_positions_B) {
  DRAKE_THROW_UNLESS(id_A.get_value() <= id_B.get_value());

  id_to_contact_edges_[id_A].insert(contact_edge_indexes_A.begin(),
                                    contact_edge_indexes_A.end());
  id_to_contact_edges_[id_B].insert(contact_edge_indexes_B.begin(),
                                    contact_edge_indexes_B.end());

  std::vector<std::tuple<T, Vector3<T>, T>> kinematic_weights_A =
      ComputeKinematicWeights(p_WCs, contact_edge_indexes_A, node_positions_A);
  std::vector<std::tuple<T, Vector3<T>, T>> kinematic_weights_B =
      ComputeKinematicWeights(p_WCs, contact_edge_indexes_B, node_positions_B);
  contact_geometry_pairs_.emplace_back(
      id_A, id_B, std::move(p_WCs), std::move(nhats_BA_W),
      std::move(signed_distances), std::move(contact_edge_indexes_A),
      std::move(kinematic_weights_A), std::move(contact_edge_indexes_B),
      std::move(kinematic_weights_B));
  if (contact_geometry_pairs_.back().num_contact_points() == 0) {
    contact_geometry_pairs_.pop_back();
  }
}

template <typename T>
void FilamentContact<T>::AddFilamentRigidContactGeometryPair(
    GeometryId id_A, GeometryId id_B, std::vector<Vector3<T>> p_WCs,
    std::vector<Vector3<T>> nhats_BA_W, std::vector<T> signed_distances,
    std::vector<int> contact_edge_indexes_A,
    const Eigen::Ref<const Eigen::Matrix3X<T>>& node_positions_A) {
  id_to_contact_edges_[id_A].insert(contact_edge_indexes_A.begin(),
                                    contact_edge_indexes_A.end());

  std::vector<std::tuple<T, Vector3<T>, T>> kinematic_weights_A =
      ComputeKinematicWeights(p_WCs, contact_edge_indexes_A, node_positions_A);
  contact_geometry_pairs_.emplace_back(
      id_A, id_B, std::move(p_WCs), std::move(nhats_BA_W),
      std::move(signed_distances), std::move(contact_edge_indexes_A),
      std::move(kinematic_weights_A));
  if (contact_geometry_pairs_.back().num_contact_points() == 0) {
    contact_geometry_pairs_.pop_back();
  }
}

template <typename T>
void FilamentContact<T>::AddFilamentFilamentContactGeometryPair(
    GeometryId id_A, GeometryId id_B,
    const std::vector<std::unique_ptr<ContactSurface<T>>>& contact_surfaces,
    const std::vector<int>& contact_edge_indexes_A_per_surface,
    const std::vector<int>& contact_edge_indexes_B_per_surface,
    const Eigen::Ref<const Eigen::Matrix3X<T>>& node_positions_A,
    const Eigen::Ref<const Eigen::Matrix3X<T>>& node_positions_B) {
  DRAKE_THROW_UNLESS(id_A.get_value() <= id_B.get_value());
  int num_contact_points = 0;
  for (const auto& contact_surface : contact_surfaces) {
    DRAKE_THROW_UNLESS(contact_surface != nullptr);
    DRAKE_THROW_UNLESS(
        (contact_surface->id_M() == id_A && contact_surface->id_N() == id_B) ||
        (contact_surface->id_M() == id_B && contact_surface->id_N() == id_A));
    DRAKE_THROW_UNLESS(contact_surface->HasGradE_M());
    DRAKE_THROW_UNLESS(contact_surface->HasGradE_N());
    DRAKE_THROW_UNLESS(contact_surface->representation() ==
                       HydroelasticContactRepresentation::kPolygon);
    num_contact_points += contact_surface->num_faces();
  }

  id_to_contact_edges_[id_A].insert(contact_edge_indexes_A_per_surface.begin(),
                                    contact_edge_indexes_A_per_surface.end());
  id_to_contact_edges_[id_B].insert(contact_edge_indexes_B_per_surface.begin(),
                                    contact_edge_indexes_B_per_surface.end());

  std::vector<Vector3<T>> p_WCs;
  std::vector<Vector3<T>> nhats_BA_W;
  std::vector<T> areas;
  std::vector<T> pressures;
  std::vector<T> signed_distances;
  std::vector<int> contact_edge_indexes_A;
  std::vector<int> contact_edge_indexes_B;
  std::vector<PolygonSurfaceMesh<T>> surface_meshes;
  p_WCs.reserve(num_contact_points);
  nhats_BA_W.reserve(num_contact_points);
  areas.reserve(num_contact_points);
  pressures.reserve(num_contact_points);
  signed_distances.reserve(num_contact_points);
  contact_edge_indexes_A.reserve(num_contact_points);
  contact_edge_indexes_B.reserve(num_contact_points);
  for (int i = 0; i < ssize(contact_surfaces); ++i) {
    const ContactSurface<T>& contact_surface = *contact_surfaces[i];
    const bool swap =
        !(contact_surface.id_M() == id_A && contact_surface.id_N() == id_B);
    for (int f = 0; f < contact_surface.num_faces(); ++f) {
      p_WCs.push_back(contact_surface.centroid(f));
      nhats_BA_W.push_back(contact_surface.face_normal(f) * (swap ? -1 : 1));
      areas.push_back(contact_surface.area(f));
      const auto [p0, phi0] =
          ComputePressureAndSignedDistance(contact_surface, f);
      pressures.push_back(p0);
      signed_distances.push_back(phi0);
      contact_edge_indexes_A.push_back(contact_edge_indexes_A_per_surface[i]);
      contact_edge_indexes_B.push_back(contact_edge_indexes_B_per_surface[i]);
    }
    surface_meshes.emplace_back(contact_surface.poly_mesh_W());
    if (swap) surface_meshes.back().ReverseFaceWinding();
  }
  std::vector<std::tuple<T, Vector3<T>, T>> kinematic_weights_A =
      ComputeKinematicWeights(p_WCs, contact_edge_indexes_A, node_positions_A);
  std::vector<std::tuple<T, Vector3<T>, T>> kinematic_weights_B =
      ComputeKinematicWeights(p_WCs, contact_edge_indexes_B, node_positions_B);
  contact_geometry_pairs_.emplace_back(
      id_A, id_B, std::move(p_WCs), std::move(nhats_BA_W),
      std::move(signed_distances), std::move(contact_edge_indexes_A),
      std::move(kinematic_weights_A), std::move(contact_edge_indexes_B),
      std::move(kinematic_weights_B), std::move(areas), std::move(pressures),
      std::move(surface_meshes));
  if (contact_geometry_pairs_.back().num_contact_points() == 0) {
    contact_geometry_pairs_.pop_back();
  }
}

template <typename T>
void FilamentContact<T>::AddFilamentRigidContactGeometryPair(
    GeometryId id_A, GeometryId id_B,
    const std::vector<std::unique_ptr<ContactSurface<T>>>& contact_surfaces,
    const std::vector<int>& contact_edge_indexes_A_per_surface,
    const Eigen::Ref<const Eigen::Matrix3X<T>>& node_positions_A) {
  int num_contact_points = 0;
  for (const auto& contact_surface : contact_surfaces) {
    DRAKE_THROW_UNLESS(contact_surface != nullptr);
    DRAKE_THROW_UNLESS(
        (contact_surface->id_M() == id_A && contact_surface->id_N() == id_B &&
         contact_surface->HasGradE_M()) ||
        (contact_surface->id_M() == id_B && contact_surface->id_N() == id_A &&
         contact_surface->HasGradE_N()));
    DRAKE_THROW_UNLESS(contact_surface->representation() ==
                       HydroelasticContactRepresentation::kPolygon);
    num_contact_points += contact_surface->num_faces();
  }

  id_to_contact_edges_[id_A].insert(contact_edge_indexes_A_per_surface.begin(),
                                    contact_edge_indexes_A_per_surface.end());

  std::vector<Vector3<T>> p_WCs;
  std::vector<Vector3<T>> nhats_BA_W;
  std::vector<T> areas;
  std::vector<T> pressures;
  std::vector<T> signed_distances;
  std::vector<int> contact_edge_indexes_A;
  std::vector<PolygonSurfaceMesh<T>> surface_meshes;
  p_WCs.reserve(num_contact_points);
  nhats_BA_W.reserve(num_contact_points);
  areas.reserve(num_contact_points);
  pressures.reserve(num_contact_points);
  signed_distances.reserve(num_contact_points);
  contact_edge_indexes_A.reserve(num_contact_points);
  for (int i = 0; i < ssize(contact_surfaces); ++i) {
    const ContactSurface<T>& contact_surface = *contact_surfaces[i];
    const bool swap =
        !(contact_surface.id_M() == id_A && contact_surface.id_N() == id_B);
    for (int f = 0; f < contact_surface.num_faces(); ++f) {
      p_WCs.push_back(contact_surface.centroid(f));
      nhats_BA_W.push_back(contact_surface.face_normal(f) * (swap ? -1 : 1));
      areas.push_back(contact_surface.area(f));
      const auto [p0, phi0] =
          ComputePressureAndSignedDistance(contact_surface, f);
      pressures.push_back(p0);
      signed_distances.push_back(phi0);
      contact_edge_indexes_A.push_back(contact_edge_indexes_A_per_surface[i]);
    }
    surface_meshes.emplace_back(contact_surface.poly_mesh_W());
    if (swap) surface_meshes.back().ReverseFaceWinding();
  }
  std::vector<std::tuple<T, Vector3<T>, T>> kinematic_weights_A =
      ComputeKinematicWeights(p_WCs, contact_edge_indexes_A, node_positions_A);
  contact_geometry_pairs_.emplace_back(
      id_A, id_B, std::move(p_WCs), std::move(nhats_BA_W),
      std::move(signed_distances), std::move(contact_edge_indexes_A),
      std::move(kinematic_weights_A), std::move(areas), std::move(pressures),
      std::move(surface_meshes));
  if (contact_geometry_pairs_.back().num_contact_points() == 0) {
    contact_geometry_pairs_.pop_back();
  }
}

template <typename T>
const std::unordered_set<int>& FilamentContact<T>::contact_edges(
    GeometryId id) const {
  const auto iter = id_to_contact_edges_.find(id);
  if (iter != id_to_contact_edges_.end()) {
    return iter->second;
  } else {
    static const std::unordered_set<int> empty_set;
    return empty_set;
  }
}

template class FilamentContactGeometryPair<double>;
template class FilamentContact<double>;

}  // namespace internal
}  // namespace geometry
}  // namespace drake
