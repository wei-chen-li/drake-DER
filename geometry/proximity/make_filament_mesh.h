#pragma once

#include <variant>

#include "drake/geometry/proximity/triangle_surface_mesh.h"
#include "drake/geometry/proximity/volume_mesh.h"
#include "drake/geometry/shape_specification.h"

namespace drake {
namespace geometry {
namespace internal {

/* Creates a volume mesh for the given `filament`.

@param filament  The filament for which a volume mesh is created.
@return The tessellated volume mesh.
@tparam_double_only
*/
template <typename T>
VolumeMesh<T> MakeFilamentVolumeMesh(const Filament& filament);

/* Creates a surface mesh for the given `filament`.

 @param filament  The filament for which a surface mesh is created.
 @return The triangulated surface.
 @tparam_double_only
*/
template <typename T>
TriangleSurfaceMesh<T> MakeFilamentSurfaceMesh(const Filament& filament);

}  // namespace internal
}  // namespace geometry
}  // namespace drake
