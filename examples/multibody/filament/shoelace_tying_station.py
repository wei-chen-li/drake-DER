import numpy as np

from pydrake.all import (
    AddCompliantHydroelasticProperties,
    AddContactMaterial,
    Box,
    BusCreator,
    Capsule,
    CollisionCheckerParams,
    CollisionFilterDeclaration,
    ConstantVectorSource,
    Convex,
    CoulombFriction,
    DeformableBodyConfig,
    Demultiplexer,
    Diagram,
    DiagramBuilder,
    DifferentialInverseKinematicsController,
    DifferentialInverseKinematicsSystem,
    DofMask,
    Filament,
    FixedOffsetFrame,
    ForceDensityField,
    GeometryInstance,
    GeometrySet,
    IllustrationProperties,
    JointLimits,
    JointStiffnessController,
    LeafSystem,
    LinearBushingRollPitchYaw,
    Mesh,
    Multiplexer,
    Parallelism,
    PassThrough,
    ProximityProperties,
    Rgba,
    RigidTransform,
    RobotDiagramBuilder,
    RotationMatrix,
    SceneGraphCollisionChecker,
    SpatialInertia,
    SpatialVelocity,
    Value,
)


class ShoelaceTyingStation(Diagram):
    def __init__(
        self,
        plant_dt=0.001,
        diff_ik_dt=0.002,
        translational_velocity_limit=[1.0, 1.0, 1.0],
        rotational_velocity_limit=[2.0, 2.0, 2.0],
        parallelism=Parallelism(False),
        **kwargs,
    ):
        super().__init__()

        robot_builder = RobotDiagramBuilder(time_step=plant_dt)
        left_arm, right_arm, _ = AddRobotArms(robot_builder)
        simplified_robot_diagram = robot_builder.Build()
        simplified_plant = simplified_robot_diagram.plant()

        robot_builder = RobotDiagramBuilder(time_step=plant_dt)
        # Add robot arms
        _, _, ground_geom_id = AddRobotArms(robot_builder)
        # Add shoe
        shoelace_bodies = AddShoe(
            robot_builder, ground_geom_id, parallelism=parallelism, **kwargs
        )

        robot_diagram = robot_builder.Build()
        plant = robot_diagram.plant()

        # Set the default configuration of the arms
        q0 = [
            0.0,
            -np.pi / 4,
            0.0,
            -3 * np.pi / 4,
            0.0,
            np.pi / 2,
            np.pi / 4,
            0.04,
            0.04,
        ] * 2
        positions0 = plant.GetDefaultPositions()
        positions0[0 : len(q0)] = q0
        plant.SetDefaultPositions(positions0)
        active_dof = DofMask(([True] * 7 + [False] * 2) * 2)

        # Configure differential inverse kinematics controller
        recipe = DifferentialInverseKinematicsSystem.Recipe()
        recipe.AddIngredient(
            DifferentialInverseKinematicsSystem.LeastSquaresCost(
                DifferentialInverseKinematicsSystem.LeastSquaresCost.Config(
                    cartesian_qp_weight=100.0
                )
            )
        )
        recipe.AddIngredient(
            DifferentialInverseKinematicsSystem.JointCenteringCost(
                DifferentialInverseKinematicsSystem.JointCenteringCost.Config(
                    posture_gain=1.0
                )
            )
        )
        JointVelocityLimitConstraint = (
            DifferentialInverseKinematicsSystem.JointVelocityLimitConstraint
        )
        recipe.AddIngredient(
            JointVelocityLimitConstraint(
                JointVelocityLimitConstraint.Config(),
                JointLimits(simplified_plant, active_dof),
            )
        )
        collision_checker = SceneGraphCollisionChecker(
            CollisionCheckerParams(
                model=simplified_robot_diagram,
                robot_model_instances=[left_arm, right_arm],
                edge_step_size=0.1,
                implicit_context_parallelism=parallelism,
            )
        )
        differential_inverse_kinematics = DifferentialInverseKinematicsSystem(
            recipe=recipe,
            task_frame=simplified_plant.world_frame().scoped_name().to_string(),
            collision_checker=collision_checker,
            active_dof=active_dof,
            time_step=diff_ik_dt,
            K_VX=1.0,
            Vd_TG_limit=SpatialVelocity(
                w=rotational_velocity_limit, v=translational_velocity_limit
            ),
        )
        diff_ik_controller = DifferentialInverseKinematicsController(
            differential_inverse_kinematics=differential_inverse_kinematics,
            planar_rotation_dof_indices=[],
        )
        diff_ik_controller.SetInitialPositions(q0)

        # Configure joint stiffness controller
        joint_stiffness_controller = JointStiffnessController(
            plant=simplified_plant,
            kp=([400] * 7 + [600] * 2) * 2,
            kd=([20] * 7 + [30] * 2) * 2,
        )

        # Connect differential inverse kinematics controller, joint stiffness
        # controller, and plant
        builder = DiagramBuilder()
        builder.AddSystem(robot_diagram)
        builder.AddSystem(diff_ik_controller)
        builder.AddSystem(joint_stiffness_controller)

        arm_state_mux = builder.AddSystem(Multiplexer([9, 9] * 2))
        demux1 = builder.AddSystem(Demultiplexer([9, 9]))
        demux2 = builder.AddSystem(Demultiplexer([9, 9]))
        builder.Connect(
            robot_diagram.GetOutputPort(
                f"plant_{plant.GetModelInstanceName(left_arm)}_state"
            ),
            demux1.get_input_port(),
        )
        builder.Connect(
            robot_diagram.GetOutputPort(
                f"plant_{plant.GetModelInstanceName(right_arm)}_state"
            ),
            demux2.get_input_port(),
        )
        builder.Connect(
            demux1.get_output_port(0), arm_state_mux.get_input_port(0)
        )
        builder.Connect(
            demux1.get_output_port(1), arm_state_mux.get_input_port(2)
        )
        builder.Connect(
            demux2.get_output_port(0), arm_state_mux.get_input_port(1)
        )
        builder.Connect(
            demux2.get_output_port(1), arm_state_mux.get_input_port(3)
        )

        builder.Connect(
            arm_state_mux.get_output_port(),
            diff_ik_controller.GetInputPort("estimated_state"),
        )
        demux1 = builder.AddSystem(Demultiplexer([7, 7]))
        demux2 = builder.AddSystem(Demultiplexer([7, 7]))
        builder.Connect(
            diff_ik_controller.GetOutputPort("commanded_position"),
            demux1.get_input_port(),
        )
        builder.Connect(
            diff_ik_controller.GetOutputPort("commanded_velocity"),
            demux2.get_input_port(),
        )

        left_hand_finger_controller = builder.AddSystem(FingerController())
        right_hand_finger_controller = builder.AddSystem(FingerController())

        builder.Connect(
            arm_state_mux.get_output_port(),
            joint_stiffness_controller.get_input_port_estimated_state(),
        )
        mux = builder.AddSystem(Multiplexer([7, 2, 7, 2] * 2))
        builder.Connect(demux1.get_output_port(0), mux.get_input_port(0))
        builder.Connect(
            left_hand_finger_controller.get_output_port(), mux.get_input_port(1)
        )
        builder.Connect(demux1.get_output_port(1), mux.get_input_port(2))
        builder.Connect(
            right_hand_finger_controller.get_output_port(),
            mux.get_input_port(3),
        )
        builder.Connect(demux2.get_output_port(0), mux.get_input_port(4))
        builder.Connect(demux2.get_output_port(1), mux.get_input_port(6))
        for i in [5, 7]:
            source = builder.AddSystem(ConstantVectorSource([0.0, 0.0]))
            builder.Connect(source.get_output_port(), mux.get_input_port(i))

        builder.Connect(
            mux.get_output_port(),
            joint_stiffness_controller.get_input_port_desired_state(),
        )
        builder.Connect(
            joint_stiffness_controller.get_output_port_actuation(),
            robot_diagram.GetInputPort("plant_actuation"),
        )

        # Connect nominal configuration input of differential inverse kinematics
        # controller
        nominal_posture_source = builder.AddSystem(ConstantVectorSource(q0))
        builder.Connect(
            nominal_posture_source.get_output_port(),
            diff_ik_controller.GetInputPort("nominal_posture"),
        )

        # Connect desired pose input of differential inverse kinematics
        # controller
        left_hand_frame = plant.GetFrameByName("panda_hand", left_arm)
        right_hand_frame = plant.GetFrameByName("panda_hand", right_arm)
        bus_creator = builder.AddSystem(BusCreator())
        model_value = Value(RigidTransform())
        bus_creator.DeclareAbstractInputPort(
            left_hand_frame.scoped_name().to_string(), model_value
        )
        bus_creator.DeclareAbstractInputPort(
            right_hand_frame.scoped_name().to_string(), model_value
        )
        builder.Connect(
            bus_creator.get_output_port(),
            diff_ik_controller.GetInputPort("desired_poses"),
        )

        # Connect pass through system to provide default poses
        plant_context = plant.CreateDefaultContext()
        X_WH_left = plant.GetFrameByName(
            "panda_hand", left_arm
        ).CalcPoseInWorld(plant_context)
        X_WH_right = plant.GetFrameByName(
            "panda_hand", right_arm
        ).CalcPoseInWorld(plant_context)
        pass_through1 = builder.AddSystem(PassThrough(Value(X_WH_left)))
        pass_through2 = builder.AddSystem(PassThrough(Value(X_WH_right)))
        builder.Connect(
            pass_through1.get_output_port(), bus_creator.get_input_port(0)
        )
        builder.Connect(
            pass_through2.get_output_port(), bus_creator.get_input_port(1)
        )

        # Export input ports
        builder.ExportInput(pass_through1.get_input_port(), "left_hand_pose")
        builder.ExportInput(pass_through2.get_input_port(), "right_hand_pose")
        builder.ExportInput(
            left_hand_finger_controller.get_input_port(), "left_hand_grasp"
        )
        builder.ExportInput(
            right_hand_finger_controller.get_input_port(), "right_hand_grasp"
        )

        # Export output ports
        builder.ExportOutput(
            robot_diagram.GetOutputPort("scene_graph_query"),
            "scene_graph_query",
        )

        # Build the diagram
        builder.BuildInto(self)
        self._q0 = q0
        self._diff_ik_controller = diff_ik_controller
        self._joint_stiffness_controller = joint_stiffness_controller
        self._robot_diagram = robot_diagram
        self._shoelace_bodies = shoelace_bodies
        self._left_arm = left_arm
        self._right_arm = right_arm

    def GetHandPoses(self, context):
        self.ValidateContext(context)
        plant = self._robot_diagram.plant()
        plant_context = self.GetSubsystemContext(plant, context)
        left_hand_frame = plant.GetFrameByName("panda_hand", self._left_arm)
        right_hand_frame = plant.GetFrameByName("panda_hand", self._right_arm)
        X_WH_left = left_hand_frame.CalcPoseInWorld(plant_context)
        X_WH_right = right_hand_frame.CalcPoseInWorld(plant_context)
        return X_WH_left, X_WH_right

    def GetDefaultHandPoses(self):
        return self.GetHandPoses(self.CreateDefaultContext())

    def GetHandGrasps(self, context):
        self.ValidateContext(context)
        plant = self._robot_diagram.plant()
        plant_context = self.GetSubsystemContext(plant, context)
        left_hand_finger1 = plant.GetJointByName(
            "panda_finger_joint1", self._left_arm
        )
        left_hand_finger2 = plant.GetJointByName(
            "panda_finger_joint2", self._left_arm
        )
        left_hand_grasp = (
            1.0
            - (
                left_hand_finger1.GetOnePosition(plant_context)
                + left_hand_finger2.GetOnePosition(plant_context)
            )
            / 0.08
        )
        right_hand_finger1 = plant.GetJointByName(
            "panda_finger_joint1", self._left_arm
        )
        right_hand_finger2 = plant.GetJointByName(
            "panda_finger_joint2", self._left_arm
        )
        right_hand_grasp = (
            1.0
            - (
                right_hand_finger1.GetOnePosition(plant_context)
                + right_hand_finger2.GetOnePosition(plant_context)
            )
            / 0.08
        )
        return left_hand_grasp, right_hand_grasp

    def GetFilamentNodePositions(self, context):
        self.ValidateContext(context)
        plant = self._robot_diagram.plant()
        plant_context = self.GetSubsystemContext(plant, context)
        return np.hstack(
            [
                filament_body.GetPositions(plant_context)
                for filament_body in self._shoelace_bodies
            ]
        )

    def GetElasticEnergy(self, context):
        self.ValidateContext(context)
        plant = self._robot_diagram.plant()
        plant_context = self.GetSubsystemContext(plant, context)
        E = 0
        for body in self._shoelace_bodies:
            E += body.CalcElasticEnergy(plant_context)
        return E


def AddRobotArms(robot_builder):
    plant = robot_builder.plant()

    # Add ground to plant
    ground = plant.AddRigidBody("ground")
    plant.WeldFrames(
        plant.world_frame(), ground.body_frame(), RigidTransform([0, 0, -0.1])
    )
    plant.RegisterVisualGeometry(
        ground, RigidTransform(), Box(10, 10, 0.2), "ground", [0.7, 0.7, 0.7, 1]
    )
    ground_geom_id = plant.RegisterCollisionGeometry(
        ground,
        RigidTransform(),
        Box(10, 10, 0.2),
        "ground",
        CoulombFriction(0.1, 0.1),
    )

    # Add robotic arm to plant
    parser = robot_builder.parser()
    parser.SetAutoRenaming(True)
    filename = "./models/panda_arm_hand.urdf"
    left_arm = parser.AddModels(filename)[0]
    right_arm = parser.AddModels(filename)[0]
    plant.WeldFrames(
        plant.world_frame(),
        plant.GetRigidBodyByName("panda_link0", left_arm).body_frame(),
        RigidTransform([-0.4, +0.3, 0]),
    )
    plant.WeldFrames(
        plant.world_frame(),
        plant.GetRigidBodyByName("panda_link0", right_arm).body_frame(),
        RigidTransform([-0.4, -0.3, 0]),
    )
    return left_arm, right_arm, ground_geom_id


def AddShoe(
    robot_builder,
    ground_geom_id,
    lace_length=0.5,
    lace_num_edges=99,
    lace_diameter=0.004,
    lace_youngs_modulus=2e4,
    lace_mass_density=50,
    lace_hydroelastic_modulus=5e4,
    lace_rgba=Rgba(0.9, 0.9, 0.9, 1.0),
    lace_as_capsule_chain=False,
    parallelism=Parallelism(False),
):
    # Add shoe to plant
    plant = robot_builder.plant()
    model_instance = plant.AddModelInstance("model")
    shoe = plant.AddRigidBody("shoe", model_instance)
    X_WS = RigidTransform(
        RotationMatrix.MakeZRotation(-np.pi / 2), [0.016, 0, 0.062]
    )
    plant.WeldFrames(plant.world_frame(), shoe.body_frame(), X_WS)
    shape = Mesh("models/shoe.gltf", 0.06)
    plant.RegisterVisualGeometry(
        shoe, RigidTransform(), shape, "shoe", [0.5, 0.5, 0.5, 1.0]
    )
    shape = Convex("models/shoe_sole.obj", 0.06)
    plant.RegisterCollisionGeometry(
        shoe, RigidTransform(), shape, "shoe_sole", CoulombFriction(0.2, 0.2)
    )
    shape = Convex("models/shoe_tongue.obj", 0.06)
    plant.RegisterCollisionGeometry(
        shoe, RigidTransform(), shape, "shoe_tongue", CoulombFriction(0.1, 0.1)
    )

    if lace_as_capsule_chain:
        collision_filter_manager = (
            robot_builder.scene_graph().collision_filter_manager()
        )

        proximity_props = ProximityProperties()
        AddProximityProperties(
            proximity_props,
            hydroelastic_modulus=lace_hydroelastic_modulus,
            circumferential_resolution_hint=lace_diameter * 0.4,
            longitudinal_resolution_hint=0.007,
            coulomb_friction=CoulombFriction(0.1, 0.1),
        )

        # Add left shoelace
        lace_edge_length = lace_length / lace_num_edges
        left_lace_model_instance = plant.AddModelInstance("left_lace")
        left_lace_geom_id = []
        left_lace_head_frame = []
        left_lace_tail_frame = []
        for i in range(lace_num_edges):
            name = f"link{i}"
            link = plant.AddRigidBody(
                name,
                left_lace_model_instance,
                SpatialInertia.SolidCapsuleWithDensity(
                    lace_mass_density,
                    lace_diameter / 2,
                    lace_edge_length,
                    [0, 0, 1],
                ),
            )
            capsule = Capsule(lace_diameter / 2, lace_edge_length)
            plant.RegisterVisualGeometry(
                link, RigidTransform(), capsule, name, lace_rgba.rgba
            )
            id = plant.RegisterCollisionGeometry(
                link, RigidTransform(), capsule, name, proximity_props
            )
            left_lace_geom_id.append(id)
            tf = RigidTransform(
                RotationMatrix.MakeXRotation(np.pi / 2),
                [0.0, 0.028 + lace_edge_length * (i + 0.5), 0.085],
            )
            if i == 0:
                plant.WeldFrames(plant.world_frame(), link.body_frame(), tf)
            else:
                plant.SetDefaultFloatingBaseBodyPose(link, tf)
            left_lace_head_frame.append(
                plant.AddFrame(
                    FixedOffsetFrame(
                        f"{name}_head_frame",
                        link,
                        RigidTransform([0, 0, lace_edge_length / 2]),
                    )
                )
            )
            left_lace_tail_frame.append(
                plant.AddFrame(
                    FixedOffsetFrame(
                        f"{name}_tail_frame",
                        link,
                        RigidTransform([0, 0, -lace_edge_length / 2]),
                    )
                )
            )

        for i in range(len(left_lace_geom_id) - 1):
            collision_filter_manager.Apply(
                CollisionFilterDeclaration().ExcludeWithin(
                    GeometrySet(
                        [left_lace_geom_id[i], left_lace_geom_id[i + 1]]
                    )
                )
            )
            plant.AddForceElement(
                LinearBushingRollPitchYaw(
                    left_lace_tail_frame[i],
                    left_lace_head_frame[i + 1],
                    torque_stiffness_constants=np.ones(3) * 1e-5,
                    torque_damping_constants=np.ones(3) * 0,
                    force_stiffness_constants=np.ones(3) * 10.0,
                    force_damping_constants=np.ones(3) * 0,
                )
            )

        # Add right shoelace
        right_lace_model_instance = plant.AddModelInstance("right_lace")
        right_lace_geom_id = []
        right_lace_head_frame = []
        right_lace_tail_frame = []
        for i in range(lace_num_edges):
            name = f"link{i}"
            link = plant.AddRigidBody(
                name,
                right_lace_model_instance,
                SpatialInertia.SolidCapsuleWithDensity(
                    lace_mass_density,
                    lace_diameter / 2,
                    lace_edge_length,
                    [0, 0, 1],
                ),
            )
            capsule = Capsule(lace_diameter / 2, lace_edge_length)
            plant.RegisterVisualGeometry(
                link, RigidTransform(), capsule, name, lace_rgba.rgba
            )
            id = plant.RegisterCollisionGeometry(
                link, RigidTransform(), capsule, name, proximity_props
            )
            right_lace_geom_id.append(id)
            tf = RigidTransform(
                RotationMatrix.MakeXRotation(-np.pi / 2),
                [0.0, -0.023 - lace_edge_length * (i + 0.5), 0.085],
            )
            if i == 0:
                plant.WeldFrames(plant.world_frame(), link.body_frame(), tf)
            else:
                plant.SetDefaultFloatingBaseBodyPose(link, tf)
            right_lace_head_frame.append(
                plant.AddFrame(
                    FixedOffsetFrame(
                        f"{name}_head_frame",
                        link,
                        RigidTransform([0, 0, lace_edge_length / 2]),
                    )
                )
            )
            right_lace_tail_frame.append(
                plant.AddFrame(
                    FixedOffsetFrame(
                        f"{name}_tail_frame",
                        link,
                        RigidTransform([0, 0, -lace_edge_length / 2]),
                    )
                )
            )

        for i in range(len(right_lace_geom_id) - 1):
            collision_filter_manager.Apply(
                CollisionFilterDeclaration().ExcludeWithin(
                    GeometrySet(
                        [right_lace_geom_id[i], right_lace_geom_id[i + 1]]
                    )
                )
            )
            plant.AddForceElement(
                LinearBushingRollPitchYaw(
                    right_lace_tail_frame[i],
                    right_lace_head_frame[i + 1],
                    torque_stiffness_constants=np.ones(3) * 1e-5,
                    torque_damping_constants=np.ones(3) * 0,
                    force_stiffness_constants=np.ones(3) * 10.0,
                    force_damping_constants=np.ones(3) * 0,
                )
            )
        return []

    # Add left shoelace
    deformable_model = robot_builder.plant().mutable_deformable_model()
    lace_edge_length = lace_length / lace_num_edges
    lace = Filament(
        closed=False,
        node_pos=np.array([[0.0, 0.0, 0.0], [0.0, lace_length, 0.0]]).T,
        cross_section=Filament.CircularCrossSection(diameter=lace_diameter),
    )
    X_WL = RigidTransform([0.0, 0.028, 0.085])
    geometry = GeometryInstance(X_WL, lace, "right_lace")

    illus_props = IllustrationProperties()
    illus_props.AddProperty("phong", "diffuse", lace_rgba)
    geometry.set_illustration_properties(illus_props)

    proximity_props = ProximityProperties()
    AddProximityProperties(
        proximity_props,
        hydroelastic_modulus=lace_hydroelastic_modulus,
        circumferential_resolution_hint=lace_diameter * 0.4,
        longitudinal_resolution_hint=0.007,
        coulomb_friction=CoulombFriction(0.1, 0.1),
    )
    geometry.set_proximity_properties(proximity_props)

    config = DeformableBodyConfig()
    config.set_youngs_modulus(lace_youngs_modulus)
    config.set_poissons_ratio(0.4999)
    config.set_mass_density(lace_mass_density)
    config.set_mass_damping_coefficient(10.0)

    left_lace_body_id = deformable_model.RegisterDeformableBody(
        geometry, config, lace_edge_length
    )
    left_lace_body = deformable_model.GetBody(left_lace_body_id)
    left_lace_geom_id = deformable_model.GetGeometryId(left_lace_body_id)

    right_lace_prop = (
        deformable_model.GetMutableBody(left_lace_body_id)
        .mutable_der_model()
        .mutable_structural_property()
    )
    right_lace_prop.set_I1(right_lace_prop.I1() * 0.5)
    right_lace_prop.set_I2(right_lace_prop.I2() * 0.5)

    deformable_model.SetWallBoundaryCondition(
        id=left_lace_body_id,
        p_WQ=X_WL @ lace.node_pos()[:, 0]
        + np.array([0, lace_edge_length * 1.001, 0]),
        n_W=[0, 1, 0],
    )

    # Add right shoelace
    lace = Filament(
        closed=False,
        node_pos=np.array([[0.0, 0.0, 0.0], [0.0, -lace_length, 0.0]]).T,
        cross_section=Filament.CircularCrossSection(diameter=lace_diameter),
    )
    X_WL = RigidTransform([0.0, -0.023, 0.085])
    geometry = GeometryInstance(X_WL, lace, "left_lace")
    geometry.set_illustration_properties(illus_props)
    geometry.set_proximity_properties(proximity_props)

    right_lace_body_id = deformable_model.RegisterDeformableBody(
        geometry, config, lace_edge_length
    )
    right_lace_body = deformable_model.GetBody(right_lace_body_id)
    right_lace_geom_id = deformable_model.GetGeometryId(right_lace_body_id)

    left_lace_prop = (
        deformable_model.GetMutableBody(right_lace_body_id)
        .mutable_der_model()
        .mutable_structural_property()
    )
    left_lace_prop.set_I1(left_lace_prop.I1() * 0.5)
    left_lace_prop.set_I2(left_lace_prop.I2() * 0.5)

    deformable_model.SetWallBoundaryCondition(
        id=right_lace_body_id,
        p_WQ=X_WL @ lace.node_pos()[:, 0]
        + np.array([0, -lace_edge_length * 1.001, 0]),
        n_W=[0, -1, 0],
    )

    # Remove the collision between the rope and the floor and add a
    # virtual field
    collision_filter_manager = (
        robot_builder.scene_graph().collision_filter_manager()
    )
    collision_filter_manager.Apply(
        CollisionFilterDeclaration().ExcludeBetween(
            GeometrySet(ground_geom_id),
            GeometrySet([left_lace_geom_id, right_lace_geom_id]),
        )
    )
    deformable_model.AddExternalForce(
        FictiousFloor(
            floor_normal=[0, 0, 1],
            floor_point=[0, 0, lace_diameter / 2],
            mass_density=lace_mass_density,
        )
    )
    deformable_model._set_parallelism(parallelism)

    return [left_lace_body, right_lace_body]


def AddProximityProperties(
    proximity_props,
    hydroelastic_modulus,
    circumferential_resolution_hint,
    longitudinal_resolution_hint,
    coulomb_friction,
    hunt_crossley_dissipation=0.01,
):
    AddCompliantHydroelasticProperties(
        properties=proximity_props,
        hydroelastic_modulus=hydroelastic_modulus,
        resolution_hint=circumferential_resolution_hint,
    )
    proximity_props.AddProperty(
        "hydroelastic",
        "circumferential_resolution_hint",
        circumferential_resolution_hint,
    )
    proximity_props.AddProperty(
        "hydroelastic",
        "longitudinal_resolution_hint",
        longitudinal_resolution_hint,
    )
    AddContactMaterial(
        properties=proximity_props,
        friction=coulomb_friction,
        dissipation=hunt_crossley_dissipation,
    )


class FingerController(LeafSystem):
    def __init__(self):
        super().__init__()
        self.DeclareVectorInputPort(name="grasp", size=1)
        self.DeclareVectorOutputPort(
            name="finger_positions", size=2, calc=self.CalcOutput
        )

    def CalcOutput(self, context, output):
        input_port = self.get_input_port()
        grasp = (
            input_port.Eval(context)[0] if input_port.HasValue(context) else 0.0
        )
        grasp = max(0.0, min(1.0, grasp))
        output.SetFromVector((1.0 - grasp) * 0.04 * np.ones(2))


class FictiousFloor(ForceDensityField):
    def __init__(
        self,
        floor_normal,
        floor_point,
        mass_density,
        stiffness=1e5,
        hunt_crossley_dissipation=0.01,
        friction_coefficient=0.1,
        K=10.0,
    ):
        super().__init__()
        self._n_WF = np.array(floor_normal, dtype=float)
        self._n_WF /= np.linalg.norm(self._n_WF)
        self._p_WF = np.array(floor_point, dtype=float)
        self._rho = mass_density
        self._k = stiffness
        self._d = hunt_crossley_dissipation
        self._mu = friction_coefficient
        self._K = K
        assert self._rho > 0
        assert self._k > 0
        assert self._d >= 0
        assert self._mu >= 0

    def DoEvaluateAt(self, context, p_WQ, v_WQ):
        return EvaluateFictiousFloorForceDensity(
            p_WQ,
            v_WQ,
            self._n_WF,
            self._p_WF,
            self._rho,
            self._k,
            self._d,
            self._mu,
            self._K,
        )

    def DoClone(self):
        return FictiousFloor(
            self._n_WF,
            self._p_WF,
            self._rho,
            self._k,
            self._d,
            self._mu,
            self._K,
        )


# @njit(fastmath=True)
def EvaluateFictiousFloorForceDensity(p_WQ, v_WQ, n_WF, p_WF, rho, k, d, mu, K):
    # Signed penetration distance (negative means penetration)
    phi = np.dot(p_WQ - p_WF, n_WF)
    # Penetration distance time derivative (negative means
    # increasing penetration)
    phi_dot = np.dot(v_WQ, n_WF)
    # Hunt–Crossley normal force
    Fn = max(0.0, -k * phi) * max(0.0, 1.0 - d * phi_dot)
    # Tangential slip velocity
    vt = v_WQ - phi_dot * n_WF
    vt_norm = np.linalg.norm(vt)
    # Sigmoid friction scaling gamma
    gamma = 2.0 / (1.0 + np.exp(-K * vt_norm)) - 1.0
    # Unit tangential direction (safe for small vt)
    epsilon = 1e-8
    if vt_norm >= epsilon:
        t_hat = vt / vt_norm
    else:
        t_hat = vt / (vt_norm + epsilon)
    # Tangential friction force
    Ft = -mu * gamma * Fn * t_hat
    # Total force density
    return (Fn * n_WF + Ft) * rho
