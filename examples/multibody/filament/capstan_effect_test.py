import numpy as np

from pydrake.all import (AddCompliantHydroelasticProperties,
                         AddContactMaterial, AddMultibodyPlant,
                         AddRigidHydroelasticProperties, Box, Capsule,
                         ConstantValueSource, ConstantVectorSource,
                         CoulombFriction, Cylinder, DeformableBodyConfig,
                         DiagramBuilder,
                         DifferentialInverseKinematicsIntegrator,
                         DifferentialInverseKinematicsParameters,
                         DiscreteDerivative, Filament, FixedOffsetFrame,
                         GeometryInstance, IllustrationProperties, Integrator,
                         LeafSystem, MeshcatVisualizer,
                         MeshcatVisualizerParams, ModelInstanceIndex,
                         MultibodyPlant, MultibodyPlantConfig, Multiplexer,
                         Parser, PdControllerGains, PidController,
                         PiecewisePolynomial, PiecewisePose,
                         PiecewiseQuaternionSlerp, PrismaticJoint,
                         ProximityProperties, Rgba, RigidTransform,
                         RotationMatrix, Saturation, SceneGraphConfig,
                         Simulator, SpatialInertia, Sphere, Value,
                         VectorLogSink)


def CapstanEffectTest(theta, delta_theta,
                      mu=0.2,
                      rope_radius=1e-3, rope_hydroelastic_modulus=1e5,
                      capstan_radius=0.03,
                      simulation_time=3.0,
                      meshcat=None):
    builder = DiagramBuilder()

    plant_config = MultibodyPlantConfig()
    plant_config.time_step = 1e-3

    plant, scene_graph = AddMultibodyPlant(plant_config, builder)

    # Add capstan
    capstan_height = capstan_radius * 2
    capstan = plant.AddRigidBody("capstan")
    plant.WeldFrames(plant.world_frame(), capstan.body_frame())
    plant.RegisterVisualGeometry(
        capstan, RigidTransform(),
        Cylinder(capstan_radius, capstan_height),
        "capstan", [0, 1, 1, 1]
    )
    proximity_props = ProximityProperties()
    AddContactMaterial(properties=proximity_props,
                       friction=CoulombFriction(mu, mu))
    AddRigidHydroelasticProperties(
        properties=proximity_props, resolution_hint=capstan_radius * 0.4)
    plant.RegisterCollisionGeometry(
        capstan, RigidTransform(),
        Cylinder(capstan_radius, capstan_height),
        "capstan", proximity_props
    )

    # Add the rope
    plant.mutable_gravity_field().set_gravity_vector([0, 0, 0])
    deformable_model = plant.mutable_deformable_model()

    r = capstan_radius + rope_radius
    delta_length = r * delta_theta
    node_positions = []
    for a in np.arange(0, theta, delta_theta):
        node_positions.append(np.array([r * np.cos(a), r * np.sin(a), 0.0]))
    for _ in range(int(0.3 * np.pi / delta_theta)):
        node_positions.insert(
            0, node_positions[0] + np.array([0.0, -delta_length, 0.0]))
        node_positions.append(
            node_positions[-1] + np.array([-np.sin(theta), np.cos(theta), 0.0]) * delta_length)

    rope = Filament(
        closed=False,
        node_pos=np.array(node_positions).T,
        cross_section=Filament.CircularCrossSection(diameter=2*rope_radius)
    )
    geometry = GeometryInstance(RigidTransform(), rope, f"rope")

    illus_props = IllustrationProperties()
    illus_props.AddProperty("phong", "diffuse", Rgba(0.6, 0.6, 0.4, 1.0))
    geometry.set_illustration_properties(illus_props)

    proximity_props = ProximityProperties()
    AddCompliantHydroelasticProperties(
        properties=proximity_props,
        hydroelastic_modulus=rope_hydroelastic_modulus,
        resolution_hint=np.nan,
    )
    proximity_props.AddProperty(
        "hydroelastic", "circumferential_resolution_hint", rope_radius * 0.4)
    proximity_props.AddProperty(
        "hydroelastic", "longitudinal_resolution_hint", delta_length)
    proximity_props.AddProperty(
        "material", "coulomb_friction", CoulombFriction(mu, mu))
    proximity_props.AddProperty("collision", "self_contact", False)
    geometry.set_proximity_properties(proximity_props)

    config = DeformableBodyConfig()
    config.set_youngs_modulus(1e5)
    config.set_poissons_ratio(0.4999)
    config.set_mass_density(10)

    unused_resolution_hint = 9999
    rope_body_id = deformable_model.RegisterDeformableBody(
        geometry, config, unused_resolution_hint)

    prop = deformable_model.GetMutableBody(
        rope_body_id).mutable_der_model().mutable_structural_property()
    prop.set_A(prop.A() * 100)

    # Add link 1
    link_radius = rope_radius * 3
    link1_model_instance = plant.AddModelInstance("link1")
    link1 = plant.AddRigidBody("link1", link1_model_instance,
                               SpatialInertia.SolidSphereWithDensity(100, link_radius))
    X_WL1 = RigidTransform(
        RotationMatrix.MakeFromOneVector(
            node_positions[1] - node_positions[0], 2),
        node_positions[0]
    )
    joint1 = plant.AddJoint(PrismaticJoint(
        name="joint1",
        frame_on_parent=plant.AddFrame(FixedOffsetFrame(
            "world_offset_frame1", plant.world_frame(), X_WL1)),
        frame_on_child=link1.body_frame(),
        axis=[0, 0, 1]
    ))
    plant.AddJointActuator("actuator1", joint1)
    plant.RegisterVisualGeometry(
        link1, RigidTransform(),
        Sphere(link_radius),
        "link1", [1, 0, 0, 1]
    )
    bbox = Sphere(delta_length * 0.1)
    deformable_model.AddFixedConstraint(
        rope_body_id, link1, X_WL1.inverse(), bbox, RigidTransform())

    # Add link 2
    link2_model_instance = plant.AddModelInstance("link2")
    link2 = plant.AddRigidBody("link2", link2_model_instance,
                               SpatialInertia.SolidSphereWithDensity(100, link_radius))
    X_WL2 = RigidTransform(
        RotationMatrix.MakeFromOneVector(
            node_positions[-2] - node_positions[-1], 2),
        node_positions[-1]
    )
    joint2 = plant.AddJoint(PrismaticJoint(
        name="joint2",
        frame_on_parent=plant.AddFrame(FixedOffsetFrame(
            "world_offset_frame2", plant.world_frame(), X_WL2)),
        frame_on_child=link2.body_frame(),
        axis=[0, 0, 1]
    ))
    plant.AddJointActuator("actuator2", joint2)
    plant.RegisterVisualGeometry(
        link2, RigidTransform(),
        Sphere(link_radius),
        "link2", [0, 0, 1, 1]
    )
    deformable_model.AddFixedConstraint(
        rope_body_id, link2, X_WL2.inverse(), bbox, RigidTransform())

    # Finalize the multibody plant
    plant.Finalize()

    # Link 1 is controlled by a PD controller with clamped output
    pid_controller = builder.AddSystem(PidController([10], [5], [0]))
    zero_source = builder.AddSystem(ConstantVectorSource([0, 0]))
    f_in = 10
    clamper = builder.AddSystem(Saturation([-f_in], [f_in]))
    builder.Connect(zero_source.get_output_port(),
                    pid_controller.get_input_port_desired_state())
    builder.Connect(plant.get_state_output_port(link1_model_instance),
                    pid_controller.get_input_port_estimated_state())
    builder.Connect(pid_controller.get_output_port(), clamper.get_input_port())
    builder.Connect(clamper.get_output_port(),
                    plant.get_actuation_input_port(link1_model_instance))

    # Link 2 in pulled by an increasing force
    constant = builder.AddSystem(ConstantVectorSource([-0.01]))
    integrator = builder.AddSystem(Integrator(1))
    builder.Connect(constant.get_output_port(), integrator.get_input_port())
    builder.Connect(integrator.get_output_port(),
                    plant.get_actuation_input_port(link2_model_instance))

    logger1 = builder.AddSystem(VectorLogSink(1))
    builder.Connect(clamper.get_output_port(), logger1.get_input_port())
    logger2 = builder.AddSystem(VectorLogSink(1))
    builder.Connect(integrator.get_output_port(), logger2.get_input_port())

    # Add Meshcat visualizer
    if meshcat:
        visualizer = MeshcatVisualizer.AddToBuilder(
            builder, scene_graph, meshcat, MeshcatVisualizerParams()
        )

    # Build and simulate
    diagram = builder.Build()
    simulator = Simulator(diagram)

    simulator.Initialize()
    simulator.AdvanceTo(simulation_time)

    log1 = logger1.FindLog(simulator.get_context())
    log2 = logger2.FindLog(simulator.get_context())
    t = log1.sample_times()
    f1 = -log1.data()[0]
    f2 = -log2.data()[0]
    return (t, f1, f2)
