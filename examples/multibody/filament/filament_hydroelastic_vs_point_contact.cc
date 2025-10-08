#include <iostream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <gflags/gflags.h>

#include "drake/geometry/meshcat.h"
#include "drake/geometry/proximity/polygon_to_triangle_mesh.h"
#include "drake/geometry/proximity_engine.h"
#include "drake/geometry/proximity_properties.h"

namespace drake {
namespace examples {
namespace filament {
namespace {

using drake::geometry::Box;
using drake::geometry::Cylinder;
using drake::geometry::Filament;
using drake::geometry::GeometryId;
using drake::geometry::Meshcat;
using drake::geometry::PolygonSurfaceMesh;
using drake::geometry::ProximityProperties;
using drake::geometry::Rgba;
using drake::geometry::Shape;
using drake::geometry::Sphere;
using drake::geometry::TriangleSurfaceMesh;
using drake::geometry::internal::FilamentContact;
using drake::geometry::internal::FilamentContactGeometryPair;
using drake::geometry::internal::MakeTriangleFromPolygonMesh;
using drake::geometry::internal::ProximityEngine;
using drake::math::RigidTransformd;
using drake::math::RotationMatrixd;
using Eigen::Vector3d;

DEFINE_bool(patch_contact, true, "Use patch contact (Default: true).");
DEFINE_string(cross_section, "circle",
              "Cross-section shape (circle or square).");
DEFINE_double(filament_width, 0.02, "Width of the filament.");
DEFINE_double(filament_segment, 0.04, "Segment length of the filament.");
DEFINE_double(sphere_radius, 0.02, "Radius of the sphere.");
DEFINE_double(sphere_position1, 0.031, "First position of the sphere.");
DEFINE_double(sphere_position2, 0.020, "Second position of the sphere.");

void DrawContacts(Meshcat* meshcat, const ProximityEngine<double>& engine,
                  std::unordered_map<GeometryId, RigidTransformd>& X_WGs,
                  double time_in_recording) {
  DRAKE_THROW_UNLESS(meshcat != nullptr);

  /* Draw patch contacts. */
  FilamentContact<double> filament_contact;
  engine.ComputeFilamentContact(&filament_contact);
  int count = 0;
  for (const FilamentContactGeometryPair<double>& pair :
       filament_contact.contact_geometry_pairs()) {
    if (!pair.is_patch_contact()) continue;
    for (const PolygonSurfaceMesh<double>& poly_mesh : pair.surface_meshes()) {
      ++count;
      const TriangleSurfaceMesh<double> tri_mesh =
          MakeTriangleFromPolygonMesh(poly_mesh);
      const std::string name = fmt::format("patch_contact/{}", count);
      meshcat->SetObject(name, tri_mesh, Rgba(1.0, 0.0, 0.0, 1.0), false, 1.0,
                         Meshcat::kDoubleSide, time_in_recording);
      meshcat->SetProperty(name, "visible", true, time_in_recording);
    }
  }
  static int max_count = 0;
  ++count;
  for (; count < max_count; ++count) {
    const std::string name = fmt::format("patch_contact/{}", count);
    meshcat->SetProperty(name, "visible", false, time_in_recording);
  }
  if (count > max_count) max_count = count;

  /* Draw point contacts. */
  auto point_pairs = engine.ComputePointPairPenetration(X_WGs);
  point_pairs.pop_back();
  for (int i = 0; i < 2; ++i) {
    const std::string name = fmt::format("point_contact/{}", i + 1);
    const Sphere point = Sphere(0.001);
    if (i < ssize(point_pairs)) {
      const Vector3d pos = (point_pairs[i].p_WCa + point_pairs[i].p_WCb) / 2;
      meshcat->SetObject(name, point, Rgba(1.0, 0.0, 0.0, 1.0),
                         time_in_recording);
      meshcat->SetTransform(name, RigidTransformd(pos), time_in_recording);
      meshcat->SetProperty(name, "visible", true, time_in_recording);
    } else {
      meshcat->SetProperty(name, "visible", false, time_in_recording);
    }
  }
}

void do_main() {
  ProximityEngine<double> engine;
  Meshcat meshcat;
  meshcat.SetProperty("/Grid", "visible", false);
  meshcat.SetProperty("/Axes", "visible", false);
  meshcat.SetProperty("/Background", "visible", false);
  meshcat.SetCameraPose(Vector3d(-0.01, -0.05, 0.06), Vector3d(0, 0, 0));

  /* Add shpere. */
  const Sphere shpere = Sphere(FLAGS_sphere_radius);
  const Vector3d sphere_t(cos(M_PI / 180 * 70), sin(M_PI / 180 * 70), 0);
  RigidTransformd X_WS(sphere_t * FLAGS_sphere_position1);
  meshcat.SetObject("sphere", shpere, Rgba(0.4, 0.5, 0.7, 0.3));
  meshcat.SetTransform("sphere", X_WS);

  const GeometryId id_S = GeometryId::get_new_id();
  ProximityProperties props;
  AddCompliantHydroelasticProperties(/* resolution_hint */ 0.005,
                                     /* hydroelastic_modulus */ 2e5, &props);
  engine.AddDynamicGeometry(shpere, X_WS, id_S, props);

  std::unordered_map<GeometryId, RigidTransformd> X_WGs;
  X_WGs[id_S] = X_WS;

  /* Parameters for filament. */
  Eigen::Matrix3Xd node_pos(3, 3);
  node_pos.col(0) = Vector3d(-1, 0, 0) * FLAGS_filament_segment;
  node_pos.col(1) = Vector3d(0, 0, 0);
  node_pos.col(2) = Vector3d(cos(M_PI / 180 * -20), sin(M_PI / 180 * -20), 0) *
                    FLAGS_filament_segment;
  Eigen::Matrix3Xd edge_m1(3, 2);
  edge_m1.col(0) = Vector3d(0, 0, -1);
  edge_m1.col(1) = Vector3d(0, 0, -1);

  /* Add geometry representing filament. */
  if (FLAGS_patch_contact) {
    /* Add deformable filament. */
    const Filament filament = [&]() {
      const double w = FLAGS_filament_width;
      if (FLAGS_cross_section == "square") {
        return Filament(
            false, node_pos, edge_m1,
            Filament::RectangularCrossSection{.width = w, .height = w});
      } else {
        return Filament(false, node_pos,
                        Filament::CircularCrossSection{.diameter = w});
      }
    }();
    meshcat.SetObject("filament", filament, Rgba(0.7, 0.5, 0.4, 0.3));

    const GeometryId id_F = GeometryId::get_new_id();
    ProximityProperties props2;
    AddCompliantHydroelasticProperties(/* resolution_hint */ 0.003,
                                       /* hydroelastic_modulus */ 1e5, &props2);
    engine.AddFilamentGeometry(filament, id_F, props2);
  } else {
    /* Add boxes or cylinders. */
    for (int i = 0; i < node_pos.cols() - 1; ++i) {
      const Vector3d c = (node_pos.col(i + 1) + node_pos.col(i)) / 2;
      const Vector3d t = (node_pos.col(i + 1) - node_pos.col(i)).normalized();
      const double l = (node_pos.col(i + 1) - node_pos.col(i)).norm();
      const double w = FLAGS_filament_width;
      const Vector3d m1 = edge_m1.col(i).normalized();
      const RigidTransformd X_WC(
          RotationMatrixd::MakeFromOrthonormalColumns(m1, t.cross(m1), t), c);
      const Box box(w, w, l);
      const Cylinder cyl(w / 2, l);
      const Shape* segment = (FLAGS_cross_section == "square")
                                 ? static_cast<const Shape*>(&box)
                                 : static_cast<const Shape*>(&cyl);
      std::string name = fmt::format("filament/cylinder{}", i);
      meshcat.SetObject(name, *segment, Rgba(0.7, 0.5, 0.4, 0.3));
      meshcat.SetTransform(name, X_WC);

      const GeometryId id_C = GeometryId::get_new_id();
      engine.AddDynamicGeometry(*segment, X_WC, id_C, {});
      X_WGs[id_C] = X_WC;
    }
  }

  /* Time and position of sphere. */
  std::vector<double> time, sphere_p;
  const int fps = 60;
  sphere_p.push_back(FLAGS_sphere_position1);
  for (int i = 1; i <= fps * 2; ++i) {
    sphere_p.push_back(i / (fps * 2.0) *
                           (FLAGS_sphere_position2 - FLAGS_sphere_position1) +
                       FLAGS_sphere_position1);
  }
  for (int i = 1; i <= fps * 2; ++i) {
    sphere_p.push_back(i / (fps * 2.0) *
                           (FLAGS_sphere_position1 - FLAGS_sphere_position2) +
                       FLAGS_sphere_position2);
  }
  for (int i = 0; i < ssize(sphere_p); ++i) {
    time.push_back(i / 60.0);
  }

  /* Record animation. */
  meshcat.StartRecording();
  for (int i = 0; i < ssize(time); ++i) {
    X_WS = RigidTransformd(sphere_t * sphere_p[i]);
    X_WGs[id_S] = X_WS;
    engine.UpdateWorldPoses(X_WGs);
    meshcat.SetTransform("sphere", X_WS, time[i]);
    DrawContacts(&meshcat, engine, X_WGs, time[i]);
  }
  meshcat.StopRecording();
  meshcat.PublishRecording();

  std::string a;
  std::cin >> a;
}

}  // namespace
}  // namespace filament
}  // namespace examples
}  // namespace drake

int main(int argc, char* argv[]) {
  gflags::SetUsageMessage("");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  drake::examples::filament::do_main();
  return 0;
}
