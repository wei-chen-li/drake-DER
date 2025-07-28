#include <memory>

#include <gflags/gflags.h>

#include "drake/geometry/drake_visualizer.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/multibody/plant/multibody_plant_config_functions.h"
#include "drake/multibody/tree/force_density_field.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/framework/diagram_builder.h"

DEFINE_double(simulation_time, 10.0, "Desired duration of the simulation [s].");
DEFINE_double(realtime_rate, 1.0, "Desired real time rate.");
DEFINE_double(time_step, 1e-2,
              "Discrete time step for the system [s]. Must be positive.");
DEFINE_double(E, 1e6, "Young's modulus of the deformable bodies [Pa].");
DEFINE_double(rho, 1000, "Mass density of the deformable bodies [kg/m³].");
DEFINE_double(diameter, 0.02, "Diameter of the cross-section [m].");
DEFINE_int32(num_edges, 200, "Number of edges.");
DEFINE_double(edge_length, 0.015, "Length of each edge [m].");
DEFINE_string(contact_approximation, "lagged",
              "Type of convex contact approximation. See "
              "multibody::DiscreteContactApproximation for details. Options "
              "are: 'sap', 'lagged', and 'similar'.");

namespace drake {
namespace examples {
namespace filament {
namespace {

using drake::geometry::Filament;
using drake::geometry::SceneGraph;
using drake::multibody::AddMultibodyPlant;
using drake::multibody::DeformableBodyId;
using drake::multibody::DeformableModel;
using drake::multibody::ForceDensityField;
using drake::multibody::ForceDensityFieldBase;
using drake::multibody::MultibodyPlant;
using drake::multibody::MultibodyPlantConfig;
using drake::multibody::fem::DeformableBodyConfig;
using drake::systems::Context;
using drake::systems::Simulator;
using Eigen::Vector3d;
using Eigen::Vector4d;
using Eigen::VectorXd;
using math::RigidTransformd;
using math::RotationMatrixd;

template <typename T>
class FictiousFloor : public ForceDensityField<T> {
 public:
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(FictiousFloor);

  FictiousFloor()
      : ForceDensityField<T>(multibody::ForceDensityType::kPerReferenceVolume) {
  }

 private:
  Vector3<T> DoEvaluateAt(const systems::Context<T>&,
                          const Vector3<T>& p_WQ) const final {
    const T phi = p_WQ[2] - FLAGS_diameter / 2;
    if (phi < 0.0)
      return Vector3<T>(0, 0, -1) * phi * FLAGS_rho * 1e4;
    else
      return Vector3<T>::Zero();
  }

  std::unique_ptr<ForceDensityFieldBase<T>> DoClone() const final {
    return std::make_unique<FictiousFloor<T>>(*this);
  }
};

DeformableBodyId RegisterFilament(DeformableModel<double>* deformable_model) {
  const bool closed = true;
  const int num_edges = FLAGS_num_edges / 4 * 4;
  const int num_edges_per_side = num_edges / 4;
  const double side_length = num_edges_per_side * FLAGS_edge_length;
  Eigen::Matrix3Xd node_pos(3, 4);
  const double a = side_length / 2;
  node_pos.col(0) = Vector3d(-a * sqrt(3) / 2, -a, -a * 1 / 2);
  node_pos.col(1) = Vector3d(+a * sqrt(3) / 2, -a, +a * 1 / 2);
  node_pos.col(2) = Vector3d(+a * sqrt(3) / 2, +a, +a * 1 / 2);
  node_pos.col(3) = Vector3d(-a * sqrt(3) / 2, +a, -a * 1 / 2);

  Filament filament(closed, node_pos,
                    Filament::CircularCrossSection{.diameter = FLAGS_diameter});

  /* Create the geometry instance from the shape shifted by z = +0.3. */
  const RigidTransformd X_WG(RotationMatrixd(), Vector3d(0, 0, 0.3));
  auto geometry_instance =
      std::make_unique<geometry::GeometryInstance>(X_WG, filament, "filament");

  /* Add a minimal illustration property for visualization. */
  geometry::IllustrationProperties illus_props;
  illus_props.AddProperty("phong", "diffuse", Vector4d(0.7, 0.5, 0.4, 1.0));
  geometry_instance->set_illustration_properties(std::move(illus_props));

  /* Add a minimal proximity property for collision detection. */
  geometry::ProximityProperties proximity_props;
  geometry_instance->set_proximity_properties(proximity_props);

  /* Set the material properties. Notice G = E / 2(1+𝜈). */
  DeformableBodyConfig<double> config;
  config.set_youngs_modulus(FLAGS_E);
  config.set_poissons_ratio(0.4999);
  config.set_mass_density(FLAGS_rho);
  config.set_mass_damping_coefficient(10.0);

  /* Add the geometry instance to the deformable model. The filament geometry is
   further discretized based on resolution_hint. */
  DeformableBodyId body_id = deformable_model->RegisterDeformableBody(
      std::move(geometry_instance), config,
      /* resolution_hint = */ FLAGS_edge_length);
  return body_id;
}

int do_main() {
  systems::DiagramBuilder<double> builder;

  MultibodyPlantConfig plant_config;
  plant_config.time_step = FLAGS_time_step;
  plant_config.discrete_contact_approximation = FLAGS_contact_approximation;

  auto [plant, scene_graph] = AddMultibodyPlant(plant_config, &builder);
  DeformableModel<double>& deformable_model = plant.mutable_deformable_model();
  RegisterFilament(&deformable_model);
  deformable_model.AddExternalForce(std::make_unique<FictiousFloor<double>>());
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
}  // namespace filament
}  // namespace examples
}  // namespace drake

int main(int argc, char* argv[]) {
  gflags::SetUsageMessage(
      "This is a demo used to showcase the undeformed curvature of a filament. "
      "The default undeformed curvature of a filament with closed ends is set "
      "to that of a circle. Therefore, even with a non-circular initial shape, "
      "the filament should eventually settle as circle. Refer to README for "
      "instructions on meldis as well as optional flags.");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  return drake::examples::filament::do_main();
}
