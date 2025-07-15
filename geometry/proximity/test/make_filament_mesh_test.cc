#include "drake/geometry/proximity/make_filament_mesh.h"

#include <optional>
#include <string>

#include <gtest/gtest.h>

#include "drake/common/eigen_types.h"
#include "drake/geometry/proximity/proximity_utilities.h"
#include "drake/geometry/proximity/triangle_surface_mesh.h"

namespace drake {
namespace geometry {
namespace internal {
namespace {

using Eigen::Vector3d;

class MakeFilamentMeshTest
    : public ::testing::TestWithParam<std::tuple<bool, std::string>> {
 private:
  void SetUp() override {
    const bool closed = std::get<0>(GetParam());
    const std::string cross_section = std::get<1>(GetParam());

    Eigen::Matrix3Xd node_pos(3, 3);
    node_pos.col(0) = Vector3d(0, 0, 0);
    node_pos.col(1) = Vector3d(1, 0, 0);
    node_pos.col(2) = Vector3d(1.0 / 2, sqrt(3) / 2, 0);

    if (cross_section == "circle") {
      filament_ = Filament(closed, node_pos,
                           Filament::CircularCrossSection{.diameter = 1e-3});
    } else {
      filament_ = Filament(
          closed, node_pos,
          Filament::RectangularCrossSection{.width = 2e-3, .height = 1e-3},
          Vector3d(0, 0, 1));
    }
  }

 protected:
  std::optional<Filament> filament_;
};

INSTANTIATE_TEST_SUITE_P(Closed_CrossSection, MakeFilamentMeshTest,
                         ::testing::Combine(::testing::Values(false, true),
                                            ::testing::Values("circle",
                                                              "rectangle")));

TEST_P(MakeFilamentMeshTest, MakeFilamentVolumeMesh) {
  MakeFilamentVolumeMesh<double>(*filament_);
}

TEST_P(MakeFilamentMeshTest, MakeFilamentSurfaceMesh) {
  MakeFilamentSurfaceMesh<double>(*filament_);
}

}  // namespace
}  // namespace internal
}  // namespace geometry
}  // namespace drake
