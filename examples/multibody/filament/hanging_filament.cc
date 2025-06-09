#include <memory>

#include <gflags/gflags.h>

#include "drake/geometry/drake_visualizer.h"
#include "drake/geometry/proximity_properties.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/multibody/plant/multibody_plant_config_functions.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/framework/diagram_builder.h"

DEFINE_double(simulation_time, 5.0, "Desired duration of the simulation [s].");
DEFINE_double(realtime_rate, 1.0, "Desired real time rate.");
DEFINE_double(time_step, 1e-3,
              "Discrete time step for the system [s]. Must be positive.");
DEFINE_double(E, 4e4, "Young's modulus of the filaments [Pa].");
DEFINE_double(G, 2e4, "Shear modulus of the filaments [Pa].");
DEFINE_double(rho, 5, "Mass density of the filaments [kg/m³].");
DEFINE_double(diameter, 0.015, "Diameter of the filaments [m].");
DEFINE_double(cylinder_diameter, 0.05, "Diameter of the cylinder [m].");
DEFINE_int32(num_edges, 101,
             "Number of edges the filaments are spatially discretized.");
DEFINE_string(contact_approximation, "lagged",
              "Type of convex contact approximation. See "
              "multibody::DiscreteContactApproximation for details. Options "
              "are: 'sap', 'lagged', and 'similar'.");

namespace drake {
namespace examples {
namespace {

using drake::geometry::Cylinder;
using drake::geometry::Filament;
using drake::geometry::SceneGraph;
using drake::geometry::Sphere;
using drake::multibody::AddMultibodyPlant;
using drake::multibody::CoulombFriction;
using drake::multibody::DeformableBodyId;
using drake::multibody::DeformableModel;
using drake::multibody::ForceDensityField;
using drake::multibody::MultibodyPlant;
using drake::multibody::MultibodyPlantConfig;
using drake::multibody::RigidBody;
using drake::multibody::SpatialInertia;
using drake::multibody::fem::DeformableBodyConfig;
using drake::systems::Context;
using drake::systems::Simulator;
using Eigen::Vector3d;
using Eigen::Vector4d;
using Eigen::VectorXd;
using math::RigidTransformd;
using math::RotationMatrixd;

DeformableBodyId RegisterFilament(DeformableModel<double>* deformable_model,
                                  const Eigen::Ref<const Vector3d>& pos0,
                                  const Eigen::Ref<const Vector3d>& pos1,
                                  bool fix_ends = false) {
  DRAKE_THROW_UNLESS(FLAGS_num_edges > 0);

  const bool closed = false;
  Eigen::Matrix3Xd node_pos(3, 2);
  node_pos.col(0) = pos0;
  node_pos.col(1) = pos1;

  Filament filament(closed, node_pos,
                    Filament::CircularCrossSection{.diameter = FLAGS_diameter});

  const RigidTransformd X_WG = RigidTransformd::Identity();
  static int i = 0;
  auto geometry_instance = std::make_unique<geometry::GeometryInstance>(
      X_WG, filament, fmt::format("filament {}", i++));

  /* Add a minimal illustration property for visualization. */
  geometry::IllustrationProperties illus_props;
  illus_props.AddProperty("phong", "diffuse", Vector4d(0.7, 0.5, 0.4, 1.0));
  geometry_instance->set_illustration_properties(std::move(illus_props));

  /* Add a minimal proximity property for collision detection. */
  geometry::ProximityProperties proximity_props;
  const CoulombFriction<double> surface_friction(0.8, 0.8);
  AddContactMaterial({}, {}, surface_friction, &proximity_props);
  geometry_instance->set_proximity_properties(proximity_props);

  /* Set the material properties. Notice G = E / 2(1+𝜈). */
  DeformableBodyConfig<double> config;
  config.set_youngs_modulus(FLAGS_E);
  config.set_poissons_ratio(0.5 * FLAGS_E / FLAGS_G - 1);
  config.set_mass_density(FLAGS_rho);
  config.set_mass_damping_coefficient(1.0);

  /* Add the geometry instance to the deformable model. The filament geometry is
   further discretized based on resolution_hint. */
  const double edge_length = (pos1 - pos0).norm() / FLAGS_num_edges;
  DeformableBodyId body_id = deformable_model->RegisterDeformableBody(
      std::move(geometry_instance), config,
      /* resolution_hint = */ edge_length);

  if (!fix_ends) return body_id;

  const Vector3d t = (pos1 - pos0).normalized();
  deformable_model->SetWallBoundaryCondition(body_id,
                                             pos0 + 0.01 * edge_length * t, t);
  deformable_model->SetWallBoundaryCondition(body_id,
                                             pos1 - 0.01 * edge_length * t, -t);
  return body_id;
}

void RegisterCylinder(MultibodyPlant<double>* plant) {
  const RigidBody<double>& cylinder = plant->AddRigidBody("cylinder");
  plant->WeldFrames(plant->world_frame(), cylinder.body_frame(),
                    RigidTransformd(RotationMatrixd::MakeXRotation(M_PI / 2),
                                    Vector3d::Zero()));
  plant->RegisterVisualGeometry(cylinder, RigidTransformd::Identity(),
                                Cylinder(FLAGS_cylinder_diameter / 2, 0.3),
                                "cylinder", Vector4d(0.2, 0.2, 0.6, 1.0));
  plant->RegisterCollisionGeometry(cylinder, RigidTransformd::Identity(),
                                   Cylinder(FLAGS_cylinder_diameter / 2, 0.3),
                                   "cylinder", CoulombFriction(0.8, 0.8));
}

int do_main() {
  systems::DiagramBuilder<double> builder;

  MultibodyPlantConfig plant_config;
  plant_config.time_step = FLAGS_time_step;
  plant_config.discrete_contact_approximation = FLAGS_contact_approximation;

  auto [plant, scene_graph] = AddMultibodyPlant(plant_config, &builder);
  DeformableModel<double>& deformable_model = plant.mutable_deformable_model();
  RegisterCylinder(&plant);
  const double z = (FLAGS_diameter / 2) + FLAGS_cylinder_diameter;
  RegisterFilament(&deformable_model, Vector3d(-0.5, 0, z),
                   Vector3d(0.5, 0, z));
  plant.Finalize();

  /* Add a visualizer that emits LCM messages for visualization. */
  geometry::DrakeVisualizerParams params;
  geometry::DrakeVisualizer<double>::AddToBuilder(&builder, scene_graph,
                                                  nullptr, params);
  auto diagram = builder.Build();

  Simulator<double> simulator(*diagram);
  simulator.Initialize();
  simulator.set_target_realtime_rate(FLAGS_realtime_rate);

  simulator.AdvanceTo(FLAGS_simulation_time);

  return 0;
}

}  // namespace
}  // namespace examples
}  // namespace drake

int main(int argc, char* argv[]) {
  gflags::SetUsageMessage(
      "This is a demo used to showcase the modeling of a cantilever beam using "
      "a deformable filament. Refer to README for instructions on meldis as "
      "well as optional flags.");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  return drake::examples::do_main();
}
