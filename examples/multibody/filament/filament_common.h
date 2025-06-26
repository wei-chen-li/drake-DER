#pragma once

#include "string_view"

#include "drake/geometry/shape_specification.h"

namespace drake {
namespace examples {
namespace filament {

geometry::Filament LoadFilament(std::string_view filament_configuration,
                                double diameter = 0.003);

}  // namespace filament
}  // namespace examples
}  // namespace drake
