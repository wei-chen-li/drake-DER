#include <array>
#include <chrono>
#include <string>
#include <thread>

#include "drake/common/text_logging.h"
#include "drake/examples/multibody/filament/filament_common.h"
#include "drake/geometry/meshcat.h"
#include "drake/geometry/proximity/polygon_to_triangle_mesh.h"
#include "drake/geometry/proximity_engine.h"
#include "drake/geometry/proximity_properties.h"

namespace drake {
namespace examples {
namespace filament {
namespace {

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
using Eigen::Vector3d;

void DrawFilamentContact(Meshcat* meshcat,
                         const ProximityEngine<double>& engine) {
  DRAKE_THROW_UNLESS(meshcat != nullptr);
  FilamentContact<double> filament_contact;
  engine.ComputeFilamentContact(&filament_contact);
  int count = 0;
  for (const FilamentContactGeometryPair<double>& pair :
       filament_contact.contact_geometry_pairs()) {
    if (!pair.is_patch_contact()) continue;
    for (const PolygonSurfaceMesh<double>& poly_mesh : pair.surface_meshes()) {
      TriangleSurfaceMesh<double> tri_mesh =
          MakeTriangleFromPolygonMesh(poly_mesh);
      meshcat->SetObject(fmt::format("contact/{}", ++count), tri_mesh,
                         Rgba(1.0, 0.5, 0.5, 1.0));
    }
  }
  static int prev_count = count;
  for (int i = count + 1; i <= prev_count; ++i)
    meshcat->Delete(fmt::format("contact/{}", i));
  prev_count = count;
}

void do_main() {
  Meshcat meshcat;
  ProximityEngine<double> engine;

  const Filament filament = LoadFilament("n3");
  meshcat.SetObject("filament", filament, Rgba(0.9, 0.9, 1.0, 0.5));

  const GeometryId id_F = GeometryId::get_new_id();
  ProximityProperties props;
  double resolution_hint = 0.003;
  double hydroelastic_modulus = 1e5;
  AddCompliantHydroelasticProperties(resolution_hint, hydroelastic_modulus,
                                     &props);
  engine.AddFilamentGeometry(filament, id_F, props);

  const Sphere shpere = Sphere(0.005);
  meshcat.SetObject("sphere", shpere, Rgba(0.9, 0.9, 0.9, 0.3));
  Vector3d p_WS = Vector3d(0.008327, 0.057763, -0.062799);

  const GeometryId id_S = GeometryId::get_new_id();
  props = ProximityProperties();
  resolution_hint = 0.005;
  hydroelastic_modulus = 1e6;
  AddCompliantHydroelasticProperties(resolution_hint, hydroelastic_modulus,
                                     &props);
  engine.AddDynamicGeometry(shpere, math::RigidTransformd(p_WS), id_S, props);

  std::array<std::string, 6> button_names = {
      "right", "left", "down", "up", "forward", "backward",
  };
  std::array<int, 6> button_clicks = {0, 0, 0, 0, 0, 0};
  meshcat.AddButton(button_names[0], "KeyD");
  meshcat.AddButton(button_names[1], "KeyA");
  meshcat.AddButton(button_names[2], "ArrowDown");
  meshcat.AddButton(button_names[3], "ArrowUp");
  meshcat.AddButton(button_names[4], "KeyW");
  meshcat.AddButton(button_names[5], "KeyS");
  meshcat.AddButton("Exit", "Escape");

  drake::log()->info(
      "Meshcat listening for connections at "
      "http://localhost:{}/?tracked_camera=on",
      meshcat.port());
  drake::log()->info("Use 'W' 'A' 'S' 'D' for control");
  drake::log()->info("Press 'Esc' to exit");

  const std::chrono::milliseconds interval(5);
  while (true) {
    auto start = std::chrono::steady_clock::now();
    if (meshcat.GetButtonClicks("Exit") > 0) break;

    std::optional<math::RigidTransformd> camera_pose =
        meshcat.GetTrackedCameraPose();
    if (!camera_pose) camera_pose = math::RigidTransformd::Identity();
    const Eigen::Matrix<double, 3, 4> X_WC = camera_pose->GetAsMatrix34();
    const double dist_CS_Cz = (p_WS - X_WC.col(3)).dot(X_WC.col(2));

    for (int i = 0; i < 6; ++i) {
      const int clicks =
          meshcat.GetButtonClicks(button_names[i]) - button_clicks[i];
      p_WS += clicks * X_WC.col(i / 2) * (i % 2 ? -1 : 1) * dist_CS_Cz * 0.01;
      button_clicks[i] += clicks;
    }
    meshcat.SetTransform("sphere", math::RigidTransformd(p_WS));
    engine.UpdateWorldPoses({{id_S, math::RigidTransformd(p_WS)}});
    DrawFilamentContact(&meshcat, engine);

    auto end = std::chrono::steady_clock::now();
    auto elapsed =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    if (elapsed < interval) std::this_thread::sleep_for(interval - elapsed);
  }
}

}  // namespace
}  // namespace filament
}  // namespace examples
}  // namespace drake

int main() {
  drake::examples::filament::do_main();
  return 0;
}
