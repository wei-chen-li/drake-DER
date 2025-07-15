#include "drake/geometry/query_results/filament_contact.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "drake/common/ssize.h"
#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/common/text_logging.h"
#include "drake/geometry/proximity_engine.h"
#include "drake/geometry/proximity_properties.h"

namespace drake {
namespace geometry {
namespace internal {
namespace {

using Eigen::Vector3d;
using testing::UnorderedElementsAre;

GTEST_TEST(FilamentPointContactTest, AddFilamentFilamentContactGeometryPair) {
  const std::vector<Vector3d> p_WCs = {Vector3d(0, 0, 0)};
  const std::vector<Vector3d> nhats_BA_W = {Vector3d(0, 0, 1)};
  const std::vector<double> signed_distances = {-1e-3};
  const std::vector<int> contact_edge_indexes_A = {0};
  const std::vector<int> contact_edge_indexes_B = {0};
  Eigen::Matrix3Xd node_positions_A(3, 2);
  node_positions_A.col(0) = Vector3d(-2, 0, -0.01);
  node_positions_A.col(1) = Vector3d(1, 0, -0.01);
  Eigen::Matrix3Xd node_positions_B(3, 2);
  node_positions_B.col(0) = Vector3d(0, -3, 0.01);
  node_positions_B.col(1) = Vector3d(0, 1, 0.01);

  const GeometryId id_A = GeometryId::get_new_id();
  const GeometryId id_B = GeometryId::get_new_id();

  FilamentContact<double> filament_contact;
  filament_contact.AddFilamentFilamentContactGeometryPair(
      id_A, id_B, p_WCs, nhats_BA_W, signed_distances, contact_edge_indexes_A,
      contact_edge_indexes_B, node_positions_A, node_positions_B);
  EXPECT_EQ(filament_contact.contact_geometry_pairs().size(), 1);
  const FilamentContactGeometryPair<double>& pair =
      filament_contact.contact_geometry_pairs()[0];
  EXPECT_EQ(pair.id_A(), id_A);
  EXPECT_EQ(pair.id_B(), id_B);
  EXPECT_TRUE(pair.is_B_filament());
  EXPECT_EQ(pair.num_contact_points(), 1);
  EXPECT_EQ(pair.R_WCs()[0].col(2), -nhats_BA_W[0]);

  constexpr double kTol = 1e-16;
  EXPECT_NEAR(std::get<0>(pair.kinematic_weights_A()[0]), 1 / 3.0, kTol);
  EXPECT_NEAR(std::get<2>(pair.kinematic_weights_A()[0]), 2 / 3.0, kTol);
  EXPECT_TRUE(CompareMatrices(std::get<1>(pair.kinematic_weights_A()[0]),
                              Vector3d(0, -0.01, 0), kTol));
  EXPECT_NEAR(std::get<0>(pair.kinematic_weights_B()[0]), 1 / 4.0, kTol);
  EXPECT_NEAR(std::get<2>(pair.kinematic_weights_B()[0]), 3 / 4.0, kTol);
  EXPECT_TRUE(CompareMatrices(std::get<1>(pair.kinematic_weights_B()[0]),
                              Vector3d(-0.01, 0, 0), kTol));

  EXPECT_THAT(filament_contact.contact_edges(id_A), UnorderedElementsAre(0));
  EXPECT_THAT(filament_contact.contact_edges(id_B), UnorderedElementsAre(0));
}

GTEST_TEST(FilamentPointContactTest, AddFilamentRigidContactGeometryPair) {
  const std::vector<Vector3d> p_WCs = {Vector3d(0, 0, 0)};
  const std::vector<Vector3d> nhats_BA_W = {Vector3d(0, 0, 1)};
  const std::vector<double> signed_distances = {-1e-3};
  const std::vector<int> contact_edge_indexes_A = {0};
  Eigen::Matrix3Xd node_positions_A(3, 2);
  node_positions_A.col(0) = Vector3d(-2, 0, -0.01);
  node_positions_A.col(1) = Vector3d(1, 0, -0.01);

  const GeometryId id_B = GeometryId::get_new_id();
  const GeometryId id_A = GeometryId::get_new_id();

  FilamentContact<double> filament_contact;
  filament_contact.AddFilamentRigidContactGeometryPair(
      id_A, id_B, p_WCs, nhats_BA_W, signed_distances, contact_edge_indexes_A,
      node_positions_A);
  EXPECT_EQ(filament_contact.contact_geometry_pairs().size(), 1);
  const FilamentContactGeometryPair<double>& pair =
      filament_contact.contact_geometry_pairs()[0];
  EXPECT_EQ(pair.id_A(), id_A);
  EXPECT_EQ(pair.id_B(), id_B);
  EXPECT_FALSE(pair.is_B_filament());
  EXPECT_EQ(pair.num_contact_points(), 1);
  EXPECT_EQ(pair.R_WCs()[0].col(2), -nhats_BA_W[0]);

  constexpr double kTol = 1e-16;
  EXPECT_NEAR(std::get<0>(pair.kinematic_weights_A()[0]), 1 / 3.0, kTol);
  EXPECT_NEAR(std::get<2>(pair.kinematic_weights_A()[0]), 2 / 3.0, kTol);
  EXPECT_TRUE(CompareMatrices(std::get<1>(pair.kinematic_weights_A()[0]),
                              Vector3d(0, -0.01, 0), kTol));

  EXPECT_THAT(filament_contact.contact_edges(id_A), UnorderedElementsAre(0));
}

class FilamentPatchContactTest : public ::testing::Test {
 protected:
  void AddSoftGeometry(const Shape& shape, const math::RigidTransformd& X_WG,
                       GeometryId id, double resolution_hint,
                       double hydroelastic_modulus = 1e6) {
    ProximityProperties props;
    AddCompliantHydroelasticProperties(resolution_hint, hydroelastic_modulus,
                                       &props);
    engine_.AddDynamicGeometry(shape, X_WG, id, props);
    X_WGs_[id] = X_WG;
  }

  void AddRigidGeometry(const Shape& shape, const math::RigidTransformd& X_WG,
                        GeometryId id, double resolution_hint) {
    ProximityProperties props;
    AddRigidHydroelasticProperties(resolution_hint, &props);
    engine_.AddDynamicGeometry(shape, X_WG, id, props);
    X_WGs_[id] = X_WG;
  }

  ContactSurface<double> ComputeContactSurface() const {
    std::vector<ContactSurface<double>> contact_surfaces =
        engine_.ComputeContactSurfaces(
            HydroelasticContactRepresentation::kPolygon, X_WGs_);
    DRAKE_THROW_UNLESS(!contact_surfaces.empty());
    return contact_surfaces[0];
  }

 private:
  ProximityEngine<double> engine_;
  std::unordered_map<GeometryId, math::RigidTransformd> X_WGs_;
};

TEST_F(FilamentPatchContactTest, AddFilamentFilamentContactGeometryPair) {
  Eigen::Matrix3Xd node_positions_A(3, 2);
  node_positions_A.col(0) = Vector3d(-0.2, 0, -0.01);
  node_positions_A.col(1) = Vector3d(0.1, 0, -0.01);
  Eigen::Matrix3Xd node_positions_B(3, 2);
  node_positions_B.col(0) = Vector3d(0.0, -3, 0.01);
  node_positions_B.col(1) = Vector3d(0.0, 1, 0.01);

  const double radius = 0.011;
  const Vector3d e_A = node_positions_A.col(1) - node_positions_A.col(0);
  const Vector3d e_B = node_positions_B.col(1) - node_positions_B.col(0);

  const GeometryId id_A = GeometryId::get_new_id();
  const GeometryId id_B = GeometryId::get_new_id();

  constexpr int kZ = 2;
  AddSoftGeometry(
      Cylinder(radius, e_A.norm()),
      math::RigidTransformd(math::RotationMatrixd::MakeFromOneVector(e_A, kZ)),
      id_A, radius * 0.5);
  AddSoftGeometry(
      Cylinder(radius, e_B.norm()),
      math::RigidTransformd(math::RotationMatrixd::MakeFromOneVector(e_B, kZ)),
      id_B, radius * 0.5);
  ContactSurface<double> contact_surface = ComputeContactSurface();
  std::vector<std::unique_ptr<ContactSurface<double>>> contact_surfaces;
  contact_surfaces.emplace_back(
      std::make_unique<ContactSurface<double>>(contact_surface));
  std::vector<int> contact_edge_indexes_A_per_surface = {0};
  std::vector<int> contact_edge_indexes_B_per_surface = {1};

  FilamentContact<double> filament_contact;
  filament_contact.AddFilamentFilamentContactGeometryPair(
      id_A, id_B, contact_surfaces, contact_edge_indexes_A_per_surface,
      contact_edge_indexes_B_per_surface, node_positions_A, node_positions_B);
  EXPECT_EQ(filament_contact.contact_geometry_pairs().size(), 1);
  const FilamentContactGeometryPair<double>& pair =
      filament_contact.contact_geometry_pairs()[0];
  EXPECT_EQ(pair.id_A(), id_A);
  EXPECT_EQ(pair.id_B(), id_B);
  EXPECT_TRUE(pair.is_B_filament());
  EXPECT_TRUE(pair.is_patch_contact());
  EXPECT_EQ(pair.num_contact_points(), contact_surface.num_faces());
  for (int i = 0; i < pair.num_contact_points(); ++i) {
    const Vector3d nhat_BA_W = contact_surface.face_normal(i) *
                               (pair.id_A() == contact_surface.id_M() ? 1 : -1);
    const double gA = contact_surface.EvaluateGradE_M_W(i).dot(nhat_BA_W);
    const double gB = contact_surface.EvaluateGradE_N_W(i).dot(-nhat_BA_W);
    DRAKE_DEMAND(gA > 0 && gB > 0);
    const double p0 =
        contact_surface.poly_e_MN().EvaluateCartesian(i, pair.p_WCs()[i]);
    const double phi0 = -p0 / (gA * gB / (gA + gB));
    DRAKE_DEMAND(p0 > 0 && phi0 < 0);

    EXPECT_EQ(pair.p_WCs()[i], contact_surface.centroid(i));
    EXPECT_EQ(pair.nhats_BA_W()[i], nhat_BA_W);
    EXPECT_EQ(pair.R_WCs()[i].col(2), -nhat_BA_W);
    EXPECT_EQ(pair.areas()[i], contact_surface.area(i));
    EXPECT_EQ(pair.pressures()[i], p0);
    EXPECT_NEAR(pair.signed_distances()[i], phi0, 1e-16);
    EXPECT_EQ(pair.contact_edge_indexes_A()[i],
              contact_edge_indexes_A_per_surface.front());
    EXPECT_EQ(pair.contact_edge_indexes_B()[i],
              contact_edge_indexes_B_per_surface.front());
  }
}

TEST_F(FilamentPatchContactTest, AddFilamentRigidContactGeometryPair) {
  Eigen::Matrix3Xd node_positions_A(3, 2);
  node_positions_A.col(0) = Vector3d(-0.2, 0, -0.01);
  node_positions_A.col(1) = Vector3d(0.1, 0, -0.01);

  const double radius = 0.011;
  const Vector3d e_A = node_positions_A.col(1) - node_positions_A.col(0);

  const GeometryId id_B = GeometryId::get_new_id();
  const GeometryId id_A = GeometryId::get_new_id();

  AddSoftGeometry(
      Cylinder(radius, e_A.norm()),
      math::RigidTransformd(math::RotationMatrixd::MakeFromOneVector(e_A, 2)),
      id_A, radius * 0.5);
  AddRigidGeometry(Sphere(radius), math::RigidTransformd(Vector3d(0, 0, 0.01)),
                   id_B, radius * 0.5);
  ContactSurface<double> contact_surface = ComputeContactSurface();
  std::vector<std::unique_ptr<ContactSurface<double>>> contact_surfaces;
  contact_surfaces.emplace_back(
      std::make_unique<ContactSurface<double>>(contact_surface));
  std::vector<int> contact_edge_indexes_A_per_surface = {0};

  FilamentContact<double> filament_contact;
  filament_contact.AddFilamentRigidContactGeometryPair(
      id_A, id_B, contact_surfaces, contact_edge_indexes_A_per_surface,
      node_positions_A);
  EXPECT_EQ(filament_contact.contact_geometry_pairs().size(), 1);
  const FilamentContactGeometryPair<double>& pair =
      filament_contact.contact_geometry_pairs()[0];
  EXPECT_EQ(pair.id_A(), id_A);
  EXPECT_EQ(pair.id_B(), id_B);
  EXPECT_FALSE(pair.is_B_filament());
  EXPECT_TRUE(pair.is_patch_contact());
  EXPECT_EQ(pair.num_contact_points(), contact_surface.num_faces());
  for (int i = 0; i < pair.num_contact_points(); ++i) {
    const Vector3d nhat_BA_W = contact_surface.face_normal(i) *
                               (pair.id_A() == contact_surface.id_M() ? 1 : -1);
    const double gA = (pair.id_A() == contact_surface.id_M())
                          ? contact_surface.EvaluateGradE_M_W(i).dot(nhat_BA_W)
                          : contact_surface.EvaluateGradE_N_W(i).dot(nhat_BA_W);
    const double p0 =
        contact_surface.poly_e_MN().EvaluateCartesian(i, pair.p_WCs()[i]);
    const double phi0 = -p0 / gA;
    DRAKE_DEMAND(p0 > 0);

    EXPECT_EQ(pair.p_WCs()[i], contact_surface.centroid(i));
    EXPECT_EQ(pair.nhats_BA_W()[i], nhat_BA_W);
    EXPECT_EQ(pair.R_WCs()[i].col(2), -nhat_BA_W);
    EXPECT_EQ(pair.areas()[i], contact_surface.area(i));
    EXPECT_EQ(pair.pressures()[i], p0);
    EXPECT_NEAR(pair.signed_distances()[i], phi0, 1e-16);
    EXPECT_EQ(pair.contact_edge_indexes_A()[i],
              contact_edge_indexes_A_per_surface.front());
  }
}

GTEST_TEST(FilamentContactTest, RemoveZeroNormals) {
  const GeometryId id_A = GeometryId::get_new_id();
  const GeometryId id_B = GeometryId::get_new_id();
  std::vector<Vector3d> p_WCs = {Vector3d(0.0, 0.0, 0.0)};
  std::vector<Vector3d> nhats_BA_W = {Vector3d::Zero()};
  std::vector<double> signed_distances = {-1e-3};
  std::vector<int> contact_edge_indexes_A = {0};
  std::vector<int> contact_edge_indexes_B = {0};
  Eigen::Matrix3Xd node_positions_A(3, 2);
  node_positions_A.col(0) = Vector3d(-1.0, 0.0, -0.01);
  node_positions_A.col(1) = Vector3d(+1.0, 0.0, -0.01);
  Eigen::Matrix3Xd node_positions_B(3, 2);
  node_positions_B.col(0) = Vector3d(0.0, -1.0, 0.01);
  node_positions_B.col(1) = Vector3d(0.0, +1.0, 0.01);

  /* Because the normal is zero, the filament_contact should remain empty. */
  FilamentContact<double> filament_contact;
  filament_contact.AddFilamentFilamentContactGeometryPair(
      id_A, id_B, p_WCs, nhats_BA_W, signed_distances, contact_edge_indexes_A,
      contact_edge_indexes_B, node_positions_A, node_positions_B);
  EXPECT_TRUE(filament_contact.contact_geometry_pairs().empty());
  filament_contact.AddFilamentRigidContactGeometryPair(
      id_A, id_B, p_WCs, nhats_BA_W, signed_distances, contact_edge_indexes_A,
      node_positions_A);
  EXPECT_TRUE(filament_contact.contact_geometry_pairs().empty());

  /* Add a contact point with nonzero normal vector. */
  p_WCs.push_back(Vector3d(0.0, 0.0, 0.0));
  nhats_BA_W.push_back(Vector3d(0.0, 0.0, -1.0));
  signed_distances.push_back(-1e-3);
  contact_edge_indexes_A.push_back(0);
  contact_edge_indexes_B.push_back(0);

  filament_contact.AddFilamentFilamentContactGeometryPair(
      id_A, id_B, p_WCs, nhats_BA_W, signed_distances, contact_edge_indexes_A,
      contact_edge_indexes_B, node_positions_A, node_positions_B);
  ASSERT_EQ(filament_contact.contact_geometry_pairs().size(), 1);
  {
    const FilamentContactGeometryPair<double>& pair =
        filament_contact.contact_geometry_pairs()[0];
    EXPECT_EQ(pair.num_contact_points(), 1);
    EXPECT_EQ(pair.p_WCs().size(), 1);
    EXPECT_EQ(pair.nhats_BA_W().size(), 1);
    EXPECT_EQ(pair.R_WCs().size(), 1);
    EXPECT_EQ(pair.signed_distances().size(), 1);
    EXPECT_EQ(pair.contact_edge_indexes_A().size(), 1);
    EXPECT_EQ(pair.kinematic_weights_A().size(), 1);
    EXPECT_EQ(pair.contact_edge_indexes_B().size(), 1);
    EXPECT_EQ(pair.kinematic_weights_B().size(), 1);
  }

  filament_contact.AddFilamentRigidContactGeometryPair(
      id_A, id_B, p_WCs, nhats_BA_W, signed_distances, contact_edge_indexes_A,
      node_positions_A);
  ASSERT_EQ(filament_contact.contact_geometry_pairs().size(), 2);
  {
    const FilamentContactGeometryPair<double>& pair =
        filament_contact.contact_geometry_pairs()[1];
    EXPECT_EQ(pair.num_contact_points(), 1);
    EXPECT_EQ(pair.p_WCs().size(), 1);
    EXPECT_EQ(pair.nhats_BA_W().size(), 1);
    EXPECT_EQ(pair.R_WCs().size(), 1);
    EXPECT_EQ(pair.signed_distances().size(), 1);
    EXPECT_EQ(pair.contact_edge_indexes_A().size(), 1);
    EXPECT_EQ(pair.kinematic_weights_A().size(), 1);
  }
}

}  // namespace
}  // namespace internal
}  // namespace geometry
}  // namespace drake
