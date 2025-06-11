#include "drake/geometry/proximity/make_filament_mesh.h"

#include <vector>

#include "drake/common/overloaded.h"
#include "drake/geometry/proximity/meshing_utilities.h"
#include "drake/geometry/proximity/volume_to_surface_mesh.h"
#include "drake/math/frame_transport.h"
#include "drake/math/rigid_transform.h"
#include "drake/math/rotation_matrix.h"

namespace drake {
namespace geometry {
namespace internal {

namespace {

std::vector<VolumeElement> SplitPolygonPrismToTetrahedra(
    int center_vertex_index0, int center_vertex_index1,
    int num_verts_per_cross_section) {
  DRAKE_THROW_UNLESS((center_vertex_index1 - center_vertex_index0) %
                         num_verts_per_cross_section ==
                     0);
  const int num_sides = num_verts_per_cross_section - 1;

  std::vector<VolumeElement> elements;
  for (int i = 0; i < num_sides; ++i) {
    const int ip1 = (i + 1) % num_sides;
    Append(SplitTriangularPrismToTetrahedra(
               center_vertex_index0, center_vertex_index0 + 1 + i,
               center_vertex_index0 + 1 + ip1, center_vertex_index1,
               center_vertex_index1 + 1 + i, center_vertex_index1 + 1 + ip1),
           &elements);
  }
  return elements;
}

template <typename T>
void Append(const Eigen::Ref<const Eigen::Matrix3X<T>>& new_vertices,
            std::vector<Vector3<T>>* mesh_vertices) {
  DRAKE_THROW_UNLESS(mesh_vertices != nullptr);
  for (int i = 0; i < new_vertices.cols(); ++i)
    mesh_vertices->emplace_back(new_vertices.col(i));
}

}  // namespace

template <typename T>
VolumeMesh<T> MakeFilamentVolumeMesh(const Filament& filament) {
  DRAKE_THROW_UNLESS(filament.edge_m1().cols() ==
                     (filament.closed() ? filament.node_pos().cols()
                                        : filament.node_pos().cols() - 1));

  /* Cross-section vertices in the cross-section frame. */
  const Eigen::Matrix3X<T> p_CVs = std::visit(
      overloaded{
          [](const Filament::RectangularCrossSection& cs) {
            Eigen::Matrix3X<T> vertices(3, 5);
            const T w = cs.width;
            const T h = cs.height;
            // clang-format off
            vertices << 0.0, -w / 2.0,  w / 2.0, w / 2.0, -w / 2.0,
                        0.0, -h / 2.0, -h / 2.0, h / 2.0,  h / 2.0,
                        0.0,      0.0,      0.0,     0.0,      0.0;
            // clang-format on
            return vertices;
          },
          [](const Filament::CircularCrossSection& cs) {
            const int kN = 20;
            Eigen::Matrix3X<T> vertices(3, kN + 1);
            auto theta =
                VectorX<T>::LinSpaced(kN + 1, 0, 2 * M_PI).head(kN).array();
            vertices.col(0).setZero();
            vertices.rightCols(kN).row(0) = 0.5 * cs.diameter * cos(theta);
            vertices.rightCols(kN).row(1) = 0.5 * cs.diameter * sin(theta);
            vertices.rightCols(kN).row(2) = VectorX<T>::Zero(kN);
            return vertices;
          }},
      filament.cross_section());
  const int num_verts_per_cross_section = p_CVs.cols();

  /* Find the tangent and m1 vector of the edge frames. */
  std::vector<Vector3<T>> edge_t;
  std::vector<Vector3<T>> edge_m1;
  std::vector<Vector3<T>> node_pos;
  node_pos.emplace_back(filament.node_pos().col(0).cast<T>());
  for (int i = 0; i < filament.edge_m1().cols(); ++i) {
    const int ip1 = (i + 1) % filament.node_pos().cols();
    const Vector3<T> edge =
        filament.node_pos().col(ip1).cast<T>() - node_pos.back();
    if (edge.norm() < 1e-10) continue;
    const Vector3<T> t = edge.normalized();
    edge_t.emplace_back(t);
    edge_m1.emplace_back(filament.edge_m1().col(i).cast<T>());
    edge_m1.back() -= edge_m1.back().dot(t) * t;
    edge_m1.back().normalize();
    if (ip1 != 0) node_pos.emplace_back(filament.node_pos().col(ip1).cast<T>());
  }
  DRAKE_DEMAND(ssize(edge_t) == ssize(edge_m1));
  DRAKE_DEMAND(ssize(edge_t) ==
               (filament.closed() ? ssize(node_pos) : ssize(node_pos) - 1));

  std::vector<VolumeElement> elements;
  std::vector<Vector3<T>> vertices;
  for (int i = 0; i < ssize(node_pos); ++i) {
    Vector3<T> t, m1;
    if (!filament.closed() && (i == 0 || i == ssize(node_pos) - 1)) {
      t = edge_t[i == 0 ? i : i - 1];
      m1 = edge_m1[i == 0 ? i : i - 1];
    } else {
      const int im1 = (i - 1 + ssize(edge_t)) % ssize(edge_t);
      t = (edge_t[im1] + edge_t[i]) / 2;
      Vector3<T> m1a, m1b;
      math::FrameTransport<T>(edge_t[im1], edge_m1[im1], t, m1a);
      math::FrameTransport<T>(edge_t[i], edge_m1[i], t, m1b);
      m1 = (m1a + m1b) / 2;
    }
    /* Pose of cross-section frame in world frame. */
    const math::RigidTransform<T> X_WC(
        math::RotationMatrix<T>::MakeFromOrthonormalColumns(m1, t.cross(m1), t),
        node_pos[i]);

    Append<double>(X_WC * p_CVs, &vertices);
    if (i == 0) continue;
    Append(SplitPolygonPrismToTetrahedra(num_verts_per_cross_section * (i - 1),
                                         num_verts_per_cross_section * i,
                                         num_verts_per_cross_section),
           &elements);
    if (filament.closed() && i == ssize(node_pos) - 1) {
      Append(SplitPolygonPrismToTetrahedra(num_verts_per_cross_section * i, 0,
                                           num_verts_per_cross_section),
             &elements);
    }
  }
  return VolumeMesh<T>(std::move(elements), std::move(vertices));
}

template <typename T>
TriangleSurfaceMesh<T> MakeFilamentSurfaceMesh(const Filament& filament) {
  return ConvertVolumeToSurfaceMesh<T>(MakeFilamentVolumeMesh<T>(filament));
}

template VolumeMesh<double> MakeFilamentVolumeMesh(const Filament&);
template TriangleSurfaceMesh<double> MakeFilamentSurfaceMesh(const Filament&);

}  // namespace internal
}  // namespace geometry
}  // namespace drake
