#include "drake/examples/multibody/filament/filament_common.h"

#include <fstream>
#include <sstream>

#include "drake/common/find_resource.h"
#include "drake/geometry/shape_specification.h"

namespace drake {
namespace examples {
namespace filament {

namespace {

Eigen::Matrix3Xd ReadPointsData(const std::string& path) {
  const std::string& filename = FindResourceOrThrow(path);
  std::ifstream infile(filename);
  if (!infile) {
    throw std::runtime_error(
        fmt::format("Failed to open file \"{}\"", filename));
  }

  std::vector<Eigen::Vector3d> points;
  std::string line;
  while (std::getline(infile, line)) {
    std::istringstream iss(line);
    double x, y, z;
    if (!(iss >> x >> y >> z)) {
      throw std::runtime_error(fmt::format("Invalid line \"{}\"", line));
    }
    points.emplace_back(x, y, z);
  }

  Eigen::Matrix3Xd mat(3, ssize(points));
  for (int i = 0; i < ssize(points); ++i) {
    mat.col(i) = points[i];
  }
  return mat;
}

}  // namespace

geometry::Filament LoadFilament(std::string_view filament_configuration,
                                double diameter) {
  const Eigen::Matrix3Xd node_positions = ReadPointsData(fmt::format(
      "drake/examples/multibody/filament/filament_configurations/{}.txt",
      filament_configuration));
  return geometry::Filament(
      false, node_positions,
      geometry::Filament::CircularCrossSection{.diameter = diameter});
}

}  // namespace filament
}  // namespace examples
}  // namespace drake
