#include <iostream>
#include <string>
#include <unordered_map>

#include <gflags/gflags.h>

#include "drake/geometry/meshcat.h"
#include "drake/geometry/proximity/polygon_to_triangle_mesh.h"
#include "drake/geometry/proximity_engine.h"
#include "drake/geometry/proximity_properties.h"

namespace drake {
namespace examples {
namespace filament {
namespace {

using drake::geometry::Cylinder;
using drake::geometry::Filament;
using drake::geometry::GeometryId;
using drake::geometry::Meshcat;
using drake::geometry::PolygonSurfaceMesh;
using drake::geometry::ProximityProperties;
using drake::geometry::Rgba;
using drake::geometry::Sphere;
using drake::geometry::TriangleSurfaceMesh;
using drake::geometry::internal::FilamentContact;
using drake::geometry::internal::FilamentContactGeometryPair;
using drake::geometry::internal::MakeTriangleFromPolygonMesh;
using drake::geometry::internal::ProximityEngine;
using drake::math::RigidTransformd;
using drake::math::RotationMatrixd;
using Eigen::Vector3d;

DEFINE_bool(point_contact, false, "Use point contact (Default: false).");

void do_main() {
  ProximityEngine<double> engine;
  Meshcat meshcat;
  meshcat.SetProperty("/Grid", "visible", false);
  meshcat.SetProperty("/Axes", "visible", false);
  meshcat.SetProperty("/Background", "visible", false);
  meshcat.SetCameraPose(Vector3d(-sqrt(1) / 2 * 0.03, sqrt(3) / 2 * 0.03, 0.09),
                        Vector3d(0, 0, 0));

  /* Add shpere. */
  const Sphere shpere = Sphere(0.02);
  meshcat.SetObject("sphere", shpere, Rgba(0.4, 0.5, 0.7, 0.3));
  RigidTransformd X_WS(Vector3d(sqrt(2) / 2, -sqrt(2) / 2, 0) * 0.02);
  meshcat.SetTransform("sphere", X_WS);

  const GeometryId id_S = GeometryId::get_new_id();
  ProximityProperties props2;
  double resolution_hint = 0.005;
  double hydroelastic_modulus = 2e5;
  AddCompliantHydroelasticProperties(resolution_hint, hydroelastic_modulus,
                                     &props2);
  engine.AddDynamicGeometry(shpere, X_WS, id_S, props2);

  /* Parameters for filament. */
  Eigen::Matrix3Xd node_positions(3, 3);
  node_positions.col(0) = Vector3d(-1, 0, 0) * 0.03;
  node_positions.col(1) = Vector3d(0, 0, 0);
  node_positions.col(2) = Vector3d(sqrt(2) / 2, sqrt(2) / 2, 0) * 0.03;
  const double diameter = 0.03;

  if (!FLAGS_point_contact) {
    /* Add filament. */
    const Filament filament(
        false, node_positions,
        Filament::CircularCrossSection{.diameter = diameter});
    meshcat.SetObject("filament", filament, Rgba(0.7, 0.5, 0.4, 0.3));

    const GeometryId id_F = GeometryId::get_new_id();
    ProximityProperties props;
    resolution_hint = 0.003;
    hydroelastic_modulus = 1e5;
    AddCompliantHydroelasticProperties(resolution_hint, hydroelastic_modulus,
                                       &props);
    engine.AddFilamentGeometry(filament, id_F, props);

    /* Draw patch contacts. */
    FilamentContact<double> filament_contact;
    engine.ComputeFilamentContact(&filament_contact);
    int count = 0;
    for (const FilamentContactGeometryPair<double>& pair :
         filament_contact.contact_geometry_pairs()) {
      if (!pair.is_patch_contact()) continue;
      for (const PolygonSurfaceMesh<double>& poly_mesh :
           pair.surface_meshes()) {
        count++;
        TriangleSurfaceMesh<double> tri_mesh =
            MakeTriangleFromPolygonMesh(poly_mesh);
        meshcat.SetObject(fmt::format("contact/{}", count), tri_mesh,
                          Rgba(1.0, 0.0, 0.0, 1.0));
      }
    }
  } else {
    /* Add cylinders. */
    std::unordered_map<GeometryId, RigidTransformd> X_WGs;
    X_WGs[id_S] = X_WS;
    for (int i = 0; i < node_positions.cols() - 1; ++i) {
      const Vector3d c =
          (node_positions.col(i + 1) + node_positions.col(i)) / 2;
      const Vector3d t = node_positions.col(i + 1) - node_positions.col(i);
      const double l = t.norm();
      const Cylinder cylinder = Cylinder(diameter / 2, l);
      std::string name = fmt::format("filament/cylinder{}", i);
      meshcat.SetObject(name, cylinder, Rgba(0.7, 0.5, 0.4, 0.3));
      RigidTransformd X_WC(RotationMatrixd::MakeFromOneVector(t, 2), c);
      meshcat.SetTransform(name, X_WC);

      const GeometryId id_C = GeometryId::get_new_id();
      engine.AddDynamicGeometry(cylinder, X_WC, id_C, {});
      X_WGs[id_C] = X_WC;
    }

    /* Draw point contacts. */
    int count = 0;
    for (auto pair : engine.ComputePointPairPenetration(X_WGs)) {
      count++;
      if (count == 3) continue;
      const Vector3d pos = (pair.p_WCa + pair.p_WCb) / 2;

      std::string name = fmt::format("contact/{}", count);
      const Sphere point = Sphere(0.001);
      meshcat.SetObject(name, point, Rgba(1.0, 0.0, 0.0, 1.0));
      meshcat.SetTransform(name, RigidTransformd(pos));
    }
  }

  std::string a;
  std::cin >> a;
}

}  // namespace
}  // namespace filament
}  // namespace examples
}  // namespace drake

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  drake::examples::filament::do_main();
  return 0;
}
