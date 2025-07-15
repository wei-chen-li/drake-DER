#include "drake/geometry/proximity/make_filament_mesh.h"

#include <algorithm>
#include <tuple>
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

template <typename T>
std::tuple<std::vector<Vector3<T>>, std::vector<Vector3<T>>,
           std::vector<Vector3<T>>>
compute_pos_t_m1(const Filament& filament) {
  const bool closed = filament.closed();
  const int num_nodes = filament.node_pos().cols();
  const int num_edges = filament.edge_m1().cols();

  const Eigen::Matrix3Xd& node_pos = filament.node_pos();
  const Eigen::Matrix3Xd& edge_m1 = filament.edge_m1();
  Eigen::Matrix3Xd edge_t(3, num_edges);
  for (int i = 0; i < num_edges; ++i) {
    const int ip1 = (i + 1) % num_nodes;
    edge_t.col(i) = (node_pos.col(ip1) - node_pos.col(i)).normalized();
  }

  std::vector<Vector3<T>> pos, t, m1;
  pos.reserve(num_nodes);
  t.reserve(num_nodes);
  m1.reserve(num_nodes);
  for (int node_index = 0; node_index < num_nodes; ++node_index) {
    pos.emplace_back(node_pos.col(node_index).cast<T>());

    if (!closed && (node_index == 0 || node_index == num_nodes - 1)) {
      const int edge_index = (node_index == 0) ? node_index : node_index - 1;
      t.emplace_back(edge_t.col(edge_index).cast<T>());
      m1.emplace_back(edge_m1.col(edge_index).cast<T>());
    } else {
      const int edge0_index = (node_index - 1 + num_edges) % num_edges;
      const int edge1_index = node_index;

      Eigen::Vector3d t_c =
          ((edge_t.col(edge0_index) + edge_t.col(edge1_index)) / 2)
              .normalized();
      t.emplace_back(t_c.cast<T>());

      Eigen::Vector3d m1_c0, m1_c1;
      math::FrameTransport<double>(edge_t.col(edge0_index),
                                   edge_m1.col(edge0_index), t.back(), m1_c0);
      math::FrameTransport<double>(edge_t.col(edge1_index),
                                   edge_m1.col(edge1_index), t.back(), m1_c1);
      Eigen::Vector3d m1_c = (m1_c0 + m1_c1) / 2;
      m1_c = (m1_c - m1_c.dot(t_c) * t_c).normalized();
      m1.emplace_back(m1_c.cast<T>());
    }
  }

  return {std::move(pos), std::move(t), std::move(m1)};
}

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

std::vector<SurfaceTriangle> SplitPolygonPrismSidefaceToTriangle(
    int start_vertex_index0, int start_vertex_index1,
    int num_verts_per_cross_section) {
  DRAKE_THROW_UNLESS((start_vertex_index1 - start_vertex_index0) %
                         num_verts_per_cross_section ==
                     0);
  const int num_sides = num_verts_per_cross_section;

  std::vector<SurfaceTriangle> triangles;
  for (int i = 0; i < num_sides; ++i) {
    const int ip1 = (i + 1) % num_sides;
    triangles.emplace_back(start_vertex_index0 + i, start_vertex_index0 + ip1,
                           start_vertex_index1 + i);
    triangles.emplace_back(start_vertex_index0 + ip1, start_vertex_index1 + ip1,
                           start_vertex_index1 + i);
  }
  return triangles;
}

template <typename T>
void AddPolygonEndcaps(std::vector<SurfaceTriangle>* triangles,
                       std::vector<Vector3<T>>* vertices,
                       int num_verts_per_cross_section) {
  DRAKE_THROW_UNLESS(triangles != nullptr);
  DRAKE_THROW_UNLESS(vertices != nullptr);
  DRAKE_THROW_UNLESS(vertices->size() % num_verts_per_cross_section == 0);
  const int num_sides = num_verts_per_cross_section;

  /* Second end cap. */
  Vector3<T> polygon_center =
      std::accumulate(vertices->rbegin(), vertices->rbegin() + num_sides,
                      Vector3<T>(0, 0, 0)) /
      num_sides;
  int offset = vertices->size() - num_verts_per_cross_section;
  vertices->push_back(polygon_center);
  for (int i = 0; i < num_sides; ++i) {
    const int ip1 = (i + 1) % num_sides;
    triangles->emplace_back(vertices->size() - 1, offset + i, offset + ip1);
  }

  /* First end cap. */
  polygon_center =
      std::accumulate(vertices->begin(), vertices->begin() + num_sides,
                      Vector3<T>(0, 0, 0)) /
      num_sides;
  offset = 0;
  vertices->push_back(polygon_center);
  for (int i = 0; i < num_sides; ++i) {
    const int ip1 = (i + 1) % num_sides;
    triangles->emplace_back(vertices->size() - 1, offset + ip1, offset + i);
  }
}

template <typename T>
void Append(const Eigen::Ref<const Eigen::Matrix3X<T>>& new_vertices,
            std::vector<Vector3<T>>* mesh_vertices) {
  DRAKE_THROW_UNLESS(mesh_vertices != nullptr);
  for (int i = 0; i < new_vertices.cols(); ++i)
    mesh_vertices->emplace_back(new_vertices.col(i));
}

void Append(const std::vector<SurfaceTriangle>& new_triangles,
            std::vector<SurfaceTriangle>* mesh_triangles) {
  DRAKE_THROW_UNLESS(mesh_triangles != nullptr);
  mesh_triangles->insert(mesh_triangles->end(), new_triangles.begin(),
                         new_triangles.end());
}

}  // namespace

template <typename T>
VolumeMesh<T> MakeFilamentVolumeMesh(const Filament& filament) {
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

  const auto [pos, t, m1] = compute_pos_t_m1<T>(filament);

  std::vector<VolumeElement> elements;
  std::vector<Vector3<T>> vertices;
  for (int i = 0; i < ssize(pos); ++i) {
    /* Pose of cross-section frame in world frame. */
    const math::RigidTransform<T> X_WC(
        math::RotationMatrix<T>::MakeFromOrthonormalColumns(
            m1[i], t[i].cross(m1[i]), t[i]),
        pos[i]);
    Append<double>(X_WC * p_CVs, &vertices);

    if (i == 0) continue;
    Append(SplitPolygonPrismToTetrahedra(num_verts_per_cross_section * (i - 1),
                                         num_verts_per_cross_section * i,
                                         num_verts_per_cross_section),
           &elements);
    if (filament.closed() && i == ssize(pos) - 1) {
      Append(SplitPolygonPrismToTetrahedra(num_verts_per_cross_section * i, 0,
                                           num_verts_per_cross_section),
             &elements);
    }
  }
  return VolumeMesh<T>(std::move(elements), std::move(vertices));
}

template <typename T>
TriangleSurfaceMesh<T> MakeFilamentSurfaceMesh(const Filament& filament) {
  /* Cross-section vertices in the cross-section frame. */
  const Eigen::Matrix3X<T> p_CVs = std::visit(
      overloaded{
          [](const Filament::RectangularCrossSection& cs) {
            Eigen::Matrix3X<T> vertices(3, 4);
            const T w = cs.width;
            const T h = cs.height;
            // clang-format off
            vertices << -w / 2.0,  w / 2.0, w / 2.0, -w / 2.0,
                        -h / 2.0, -h / 2.0, h / 2.0,  h / 2.0,
                             0.0,      0.0,     0.0,      0.0;
            // clang-format on
            return vertices;
          },
          [](const Filament::CircularCrossSection& cs) {
            const int kN = 20;
            Eigen::Matrix3X<T> vertices(3, kN);
            auto theta =
                VectorX<T>::LinSpaced(kN + 1, 0, 2 * M_PI).head(kN).array();
            vertices.row(0) = 0.5 * cs.diameter * cos(theta);
            vertices.row(1) = 0.5 * cs.diameter * sin(theta);
            vertices.row(2) = VectorX<T>::Zero(kN);
            return vertices;
          }},
      filament.cross_section());
  const int num_verts_per_cross_section = p_CVs.cols();

  const auto [pos, t, m1] = compute_pos_t_m1<T>(filament);

  std::vector<SurfaceTriangle> triangles;
  std::vector<Vector3<T>> vertices;
  for (int i = 0; i < ssize(pos); ++i) {
    /* Pose of cross-section frame in world frame. */
    const math::RigidTransform<T> X_WC(
        math::RotationMatrix<T>::MakeFromOrthonormalColumns(
            m1[i], t[i].cross(m1[i]), t[i]),
        pos[i]);
    Append<double>(X_WC * p_CVs, &vertices);

    if (i == 0) continue;
    Append(SplitPolygonPrismSidefaceToTriangle(
               num_verts_per_cross_section * (i - 1),
               num_verts_per_cross_section * i, num_verts_per_cross_section),
           &triangles);
    if (filament.closed() && i == ssize(pos) - 1) {
      Append(
          SplitPolygonPrismSidefaceToTriangle(num_verts_per_cross_section * i,
                                              0, num_verts_per_cross_section),
          &triangles);
    }
  }
  if (!filament.closed()) {
    AddPolygonEndcaps(&triangles, &vertices, num_verts_per_cross_section);
  }
  return TriangleSurfaceMesh<T>(std::move(triangles), std::move(vertices));
}

template VolumeMesh<double> MakeFilamentVolumeMesh(const Filament&);
template TriangleSurfaceMesh<double> MakeFilamentSurfaceMesh(const Filament&);

}  // namespace internal
}  // namespace geometry
}  // namespace drake
