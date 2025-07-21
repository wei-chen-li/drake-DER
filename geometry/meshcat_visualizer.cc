#include "drake/geometry/meshcat_visualizer.h"

#include <algorithm>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include <fmt/format.h>

#include "drake/common/extract_double.h"
#include "drake/common/overloaded.h"
#include "drake/geometry/meshcat_graphviz.h"
#include "drake/geometry/meshcat_internal.h"
#include "drake/geometry/proximity/polygon_to_triangle_mesh.h"
#include "drake/geometry/proximity/volume_to_surface_mesh.h"
#include "drake/geometry/utilities.h"

namespace drake {
namespace geometry {

template <typename T>
MeshcatVisualizer<T>::MeshcatVisualizer(std::shared_ptr<Meshcat> meshcat,
                                        MeshcatVisualizerParams params)
    : systems::LeafSystem<T>(systems::SystemTypeTag<MeshcatVisualizer>{}),
      meshcat_(std::move(meshcat)),
      params_(std::move(params)),
      alpha_slider_name_(std::string(params_.prefix + " α")) {
  DRAKE_DEMAND(meshcat_ != nullptr);
  DRAKE_DEMAND(params_.publish_period >= 0.0);
  if (params_.role == Role::kUnassigned) {
    throw std::runtime_error(
        "MeshcatVisualizer cannot be used for geometries with the "
        "Role::kUnassigned value. Please choose kProximity, kPerception, or "
        "kIllustration");
  }

  this->DeclarePeriodicPublishEvent(params_.publish_period, 0.0,
                                    &MeshcatVisualizer<T>::UpdateMeshcat);
  this->DeclareForcedPublishEvent(&MeshcatVisualizer<T>::UpdateMeshcat);

  if (params_.delete_on_initialization_event) {
    this->DeclareInitializationPublishEvent(
        &MeshcatVisualizer<T>::OnInitialization);
  }

  query_object_input_port_ =
      this->DeclareAbstractInputPort("query_object", Value<QueryObject<T>>())
          .get_index();

  if (params_.enable_alpha_slider) {
    alpha_value_ = params_.initial_alpha_slider_value;
    meshcat_->AddSlider(alpha_slider_name_, 0.02, 1.0, 0.02, alpha_value_);
  }
}

template <typename T>
template <typename U>
MeshcatVisualizer<T>::MeshcatVisualizer(const MeshcatVisualizer<U>& other)
    : MeshcatVisualizer(other.meshcat_, other.params_) {}

template <typename T>
MeshcatVisualizer<T>::~MeshcatVisualizer() = default;

template <typename T>
void MeshcatVisualizer<T>::Delete() const {
  meshcat_->Delete(params_.prefix);
  version_ = std::nullopt;
}

template <typename T>
MeshcatAnimation* MeshcatVisualizer<T>::StartRecording(
    bool set_transforms_while_recording) {
  meshcat_->StartRecording(1.0 / params_.publish_period,
                           set_transforms_while_recording);
  return &meshcat_->get_mutable_recording();
}

template <typename T>
void MeshcatVisualizer<T>::StopRecording() {
  meshcat_->StopRecording();
}

template <typename T>
void MeshcatVisualizer<T>::PublishRecording() const {
  meshcat_->PublishRecording();
}

template <typename T>
void MeshcatVisualizer<T>::DeleteRecording() {
  meshcat_->DeleteRecording();
}

template <typename T>
MeshcatAnimation* MeshcatVisualizer<T>::get_mutable_recording() {
  return &meshcat_->get_mutable_recording();
}

template <typename T>
MeshcatVisualizer<T>& MeshcatVisualizer<T>::AddToBuilder(
    systems::DiagramBuilder<T>* builder, const SceneGraph<T>& scene_graph,
    std::shared_ptr<Meshcat> meshcat, MeshcatVisualizerParams params) {
  return AddToBuilder(builder, scene_graph.get_query_output_port(),
                      std::move(meshcat), std::move(params));
}

template <typename T>
MeshcatVisualizer<T>& MeshcatVisualizer<T>::AddToBuilder(
    systems::DiagramBuilder<T>* builder,
    const systems::OutputPort<T>& query_object_port,
    std::shared_ptr<Meshcat> meshcat, MeshcatVisualizerParams params) {
  const std::string aspirational_name =
      fmt::format("meshcat_visualizer({})", params.prefix);
  auto& visualizer = *builder->template AddSystem<MeshcatVisualizer<T>>(
      std::move(meshcat), std::move(params));
  if (!builder->HasSubsystemNamed(aspirational_name)) {
    visualizer.set_name(aspirational_name);
  }
  builder->Connect(query_object_port, visualizer.query_object_input_port());
  return visualizer;
}

template <typename T>
systems::EventStatus MeshcatVisualizer<T>::UpdateMeshcat(
    const systems::Context<T>& context) const {
  const auto& query_object =
      query_object_input_port().template Eval<QueryObject<T>>(context);
  const GeometryVersion& current_version =
      query_object.inspector().geometry_version();
  if (!version_.has_value()) {
    // When our current version is null, that means we haven't added any
    // geometry to Meshcat yet, which means we also need to establish our
    // default visibility just prior to sending the geometry.
    meshcat_->SetProperty(params_.prefix, "visible",
                          params_.visible_by_default);
  }
  if (!version_.has_value() ||
      !version_->IsSameAs(current_version, params_.role)) {
    SetObjects(query_object.inspector());
    SetAlphas(/* initializing = */ true);
    version_ = current_version;
  }
  SetTransforms(context, query_object);
  SetDeformables(context, query_object);
  if (params_.enable_alpha_slider) {
    double new_alpha_value = meshcat_->GetSliderValue(alpha_slider_name_);
    if (new_alpha_value != alpha_value_) {
      alpha_value_ = new_alpha_value;
      SetAlphas(/* initializing = */ false);
    }
  }
  meshcat_->SetSimulationTime(ExtractDoubleOrThrow(context.get_time()));

  return systems::EventStatus::Succeeded();
}

template <typename T>
void MeshcatVisualizer<T>::SetObjects(
    const SceneGraphInspector<T>& inspector) const {
  // Frames registered previously that are not set again here should be deleted.
  std::map<FrameId, std::string> frames_to_delete{};
  dynamic_frames_.swap(frames_to_delete);

  // Geometries registered previously that are not set again here should be
  // deleted.
  std::map<GeometryId, std::string> geometries_to_delete{};
  geometries_.swap(geometries_to_delete);

  // TODO(SeanCurtis-TRI): Mimic the full tree structure in SceneGraph.
  // SceneGraph supports arbitrary hierarchies of frames just like Meshcat.
  // This code is arbitrarily flattening it because the current SceneGraph API
  // is insufficient to support walking the tree.
  for (FrameId frame_id : inspector.GetAllFrameIds()) {
    std::string frame_path =
        frame_id == inspector.world_frame_id()
            ? params_.prefix
            : fmt::format("{}/{}", params_.prefix, inspector.GetName(frame_id));
    // MultibodyPlant declares frames with SceneGraph using "::". We replace
    // those with `/` here to expose the full tree to Meshcat.
    size_t pos = 0;
    while ((pos = frame_path.find("::", pos)) != std::string::npos) {
      frame_path.replace(pos++, 2, "/");
    }

    bool frame_has_any_geometry = false;
    for (GeometryId geom_id : inspector.GetGeometries(frame_id, params_.role)) {
      const GeometryProperties& properties =
          *inspector.GetProperties(geom_id, params_.role);
      if (properties.HasProperty("meshcat", "accepting")) {
        if (properties.GetProperty<std::string>("meshcat", "accepting") !=
            params_.prefix) {
          continue;
        }
      } else if (!params_.include_unspecified_accepting) {
        continue;
      }

      // We'll turn scoped names into meshcat paths.
      const std::string geometry_name =
          internal::TransformGeometryName(geom_id, inspector);
      const std::string path = fmt::format("{}/{}", frame_path, geometry_name);
      const Rgba rgba = properties.GetPropertyOrDefault("phong", "diffuse",
                                                        params_.default_color);

      // The "object" will typically be the geometry's shape. But, for the
      // proximity role, we prefer, first, the hydroelastic surface if
      // available, or, second, the convex hull. Record if we've used one of
      // those proxies.
      bool geometry_already_set = false;

      if constexpr (std::is_same_v<T, double>) {
        if (params_.show_hydroelastic) {
          auto maybe_mesh = inspector.maybe_get_hydroelastic_mesh(geom_id);
          std::visit<void>(
              overloaded{[](std::monostate) {},
                         [&](const TriangleSurfaceMesh<double>* mesh) {
                           DRAKE_DEMAND(mesh != nullptr);
                           meshcat_->SetObject(path, *mesh, rgba);
                           geometry_already_set = true;
                         },
                         [&](const VolumeMesh<double>* mesh) {
                           DRAKE_DEMAND(mesh != nullptr);
                           meshcat_->SetObject(
                               path, ConvertVolumeToSurfaceMesh(*mesh), rgba);
                           geometry_already_set = true;
                         }},
              maybe_mesh);
        }
      }

      // Proximity role favors convex hulls if available.
      if (const PolygonSurfaceMesh<double>* hull = nullptr;
          (!geometry_already_set) && (params_.role == Role::kProximity) &&
          (hull = inspector.GetConvexHull(geom_id))) {
        // Convert polygonal surface mesh to triangle surface mesh.
        const TriangleSurfaceMesh<double> tri_hull =
            internal::MakeTriangleFromPolygonMesh(*hull);
        meshcat_->SetObject(path, tri_hull, rgba);
        geometry_already_set = true;
      }

      if (!geometry_already_set) {
        meshcat_->SetObject(path, inspector.GetShape(geom_id), rgba);
      }

      meshcat_->SetTransform(path, inspector.GetPoseInFrame(geom_id));
      geometries_[geom_id] = path;
      geometries_to_delete.erase(geom_id);  // Don't delete this one.
      frame_has_any_geometry = true;
    }

    if (frame_has_any_geometry && (frame_id != inspector.world_frame_id())) {
      dynamic_frames_[frame_id] = frame_path;
      frames_to_delete.erase(frame_id);  // Don't delete this one.
    }
  }

  for (const auto& [geom_id, path] : geometries_to_delete) {
    unused(geom_id);
    meshcat_->Delete(path);
  }
  for (const auto& [frame_id, path] : frames_to_delete) {
    unused(frame_id);
    meshcat_->Delete(path);
  }
}

template <typename T>
void MeshcatVisualizer<T>::SetTransforms(
    const systems::Context<T>& context,
    const QueryObject<T>& query_object) const {
  for (const auto& [frame_id, path] : dynamic_frames_) {
    const math::RigidTransformd X_WF =
        internal::convert_to_double(query_object.GetPoseInWorld(frame_id));
    meshcat_->SetTransform(path, X_WF,
                           ExtractDoubleOrThrow(context.get_time()));
  }
}

template <typename T>
void MeshcatVisualizer<T>::SetAlphas(bool initializing) const {
  if (initializing) {
    for (const auto& [_, geo_path] : geometries_) {
      meshcat_->SetProperty(geo_path, "modulated_opacity", alpha_value_);
    }
  } else {
    // The geometries visualized by this visualizer (stored in geometries_) all
    // have a common prefix and for a well-configured visualizer, it is a
    // *unique* prefix. So, we can rely on meshcat's behavior to update all
    // materials in a tree with a single invocation on the root path. This
    // requires that all object instantiations are complete in the visualizer
    // instance.
    meshcat_->SetProperty(params_.prefix, "modulated_opacity", alpha_value_);
  }
}

template <typename T>
void MeshcatVisualizer<T>::SetDeformables(
    const systems::Context<T>& context,
    const QueryObject<T>& query_object) const {
  const SceneGraphInspector<T>& inspector = query_object.inspector();
  for (const GeometryId geom_id : inspector.GetAllDeformableGeometryIds()) {
    // If the geometry doesn't have the role that the visualizer wants to
    // visualize, skip it.
    if (inspector.GetProperties(geom_id, params_.role) == nullptr) {
      continue;
    }
    // For a given geometry, if the property (meshcat, accepting) exists then
    // the visualizer will show the geometry only if the property's value
    // matches our prefix. If that property is absent then the geometry will be
    // shown only if include_unspecified_accepting is true.
    const GeometryProperties& properties =
        *inspector.GetProperties(geom_id, params_.role);
    if (properties.HasProperty("meshcat", "accepting")) {
      if (properties.GetProperty<std::string>("meshcat", "accepting") !=
          params_.prefix) {
        continue;
      }
    } else if (!params_.include_unspecified_accepting) {
      continue;
    }

    // We'll turn scoped names into meshcat paths.
    const std::string geometry_name =
        internal::TransformGeometryName(geom_id, inspector);
    const std::string path =
        fmt::format("{}/{}", params_.prefix, geometry_name);

    const VolumeMesh<double>* reference_mesh =
        inspector.GetReferenceMesh(geom_id);
    const Filament* reference_filament =
        inspector.GetReferenceFilament(geom_id);
    if (reference_mesh == nullptr && reference_filament == nullptr) continue;

    if (reference_mesh != nullptr) {
      // Get the render mesh and mesh vertices positions.
      const std::vector<internal::RenderMesh>& render_meshes =
          inspector.GetDrivenRenderMeshes(geom_id, params_.role);
      const std::vector<VectorX<T>> vertex_positions =
          query_object.GetDrivenMeshConfigurationsInWorld(geom_id,
                                                          params_.role);
      DRAKE_DEMAND(ssize(vertex_positions) == ssize(render_meshes));
      if (render_meshes.empty()) continue;

      // There is typically only one render mesh and its associated vertex
      // positions.
      const Eigen::VectorXd vertices =
          ExtractDoubleOrThrow(vertex_positions[0]);
      const Eigen::Matrix<unsigned int, Eigen::Dynamic, 3, Eigen::RowMajor>&
          faces = render_meshes[0].indices;

      const Rgba& rgba = render_meshes[0].material.has_value()
                             ? render_meshes[0].material->diffuse
                             : params_.default_color;

      meshcat_->SetTransform(path, math::RigidTransformd());
      meshcat_->SetTriangleMesh(path,
                                Eigen::Map<const Eigen::Matrix3Xd>(
                                    vertices.data(), 3, ssize(vertices) / 3),
                                faces.cast<int>().transpose(), rgba,
                                /* wireframe = */ false,
                                /* wireframe_line_width = */ 1.0,
                                Meshcat::kDoubleSide,
                                ExtractDoubleOrThrow(context.get_time()));
    } else {
      const bool closed = reference_filament->closed();
      const int num_nodes = reference_filament->node_pos().cols();
      const int num_edges = reference_filament->edge_m1().cols();

      const VectorX<double> q_WF =
          ExtractDoubleOrThrow(query_object.GetConfigurationsInWorld(geom_id));
      DRAKE_DEMAND(q_WF.size() == 3 * num_nodes + 3 * num_edges);
      const Eigen::Matrix3Xd node_pos =
          Eigen::Map<const Eigen::Matrix3Xd>(q_WF.data(), 3, num_nodes);
      const Eigen::Matrix3Xd edge_m1 = Eigen::Map<const Eigen::Matrix3Xd>(
          q_WF.data() + 3 * num_nodes, 3, num_edges);
      const Filament filament(closed, node_pos, edge_m1,
                              reference_filament->cross_section());

      const Rgba rgba = properties.GetPropertyOrDefault("phong", "diffuse",
                                                        params_.default_color);

      meshcat_->SetTransform(path, math::RigidTransformd());
      meshcat_->SetObject(path, filament, rgba);
    }
    geometries_[geom_id] = path;
  }
}

template <typename T>
systems::EventStatus MeshcatVisualizer<T>::OnInitialization(
    const systems::Context<T>&) const {
  Delete();
  return systems::EventStatus::Succeeded();
}

template <typename T>
typename systems::LeafSystem<T>::GraphvizFragment
MeshcatVisualizer<T>::DoGetGraphvizFragment(
    const typename systems::LeafSystem<T>::GraphvizFragmentParams& params)
    const {
  internal::MeshcatGraphviz meshcat_graphviz(params_.prefix,
                                             /* subscribe = */ false);
  return meshcat_graphviz.DecorateResult(
      systems::LeafSystem<T>::DoGetGraphvizFragment(
          meshcat_graphviz.DecorateParams(params)));
}

}  // namespace geometry
}  // namespace drake

DRAKE_DEFINE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_NONSYMBOLIC_SCALARS(
    class ::drake::geometry::MeshcatVisualizer);
