#include "drake/bindings/pydrake/common/cpp_template_pybind.h"
#include "drake/bindings/pydrake/common/default_scalars_pybind.h"
#include "drake/bindings/pydrake/common/type_pack.h"
#include "drake/bindings/pydrake/documentation_pybind.h"
#include "drake/bindings/pydrake/pydrake_pybind.h"
#include "drake/common/default_scalars.h"
#include "drake/multibody/der/der_model.h"

namespace drake {
namespace pydrake {
namespace {

template <typename T>
void DoScalarDependentDefinitions(py::module m, T) {
  py::tuple param = GetPyParam<T>();

  // NOLINTNEXTLINE(build/namespaces): Emulate placement in namespace.
  using namespace drake::multibody::der;
  constexpr auto& doc = pydrake_doc.drake.multibody.der;

  {
    using Class = DerStructuralProperty<T>;
    constexpr auto& cls_doc = doc.DerStructuralProperty;
    auto cls = DefineTemplateClassWithDefault<Class>(
        m, "DerStructuralProperty", param, cls_doc.doc);
    cls  // BR
        .def_static("FromRectangularCrossSection",
            &Class::FromRectangularCrossSection, py::arg("width"),
            py::arg("height"), py::arg("youngs_modulus"),
            py::arg("shear_modulus"), py::arg("mass_density"),
            cls_doc.FromRectangularCrossSection.doc)
        .def_static("FromEllipticalCrossSection",
            &Class::FromEllipticalCrossSection, py::arg("a"), py::arg("b"),
            py::arg("youngs_modulus"), py::arg("shear_modulus"),
            py::arg("mass_density"), cls_doc.FromEllipticalCrossSection.doc)
        .def_static("FromCircularCrossSection",
            &Class::FromCircularCrossSection, py::arg("r"),
            py::arg("youngs_modulus"), py::arg("shear_modulus"),
            py::arg("mass_density"), cls_doc.FromCircularCrossSection.doc)
        .def("A", &Class::A, cls_doc.A.doc)
        .def("I1", &Class::I1, cls_doc.I1.doc)
        .def("I2", &Class::I2, cls_doc.I2.doc)
        .def("EA", &Class::EA, cls_doc.EA.doc)
        .def("EI1", &Class::EI1, cls_doc.EI1.doc)
        .def("EI2", &Class::EI2, cls_doc.EI2.doc)
        .def("GJ", &Class::GJ, cls_doc.GJ.doc)
        .def("rhoA", &Class::rhoA, cls_doc.rhoA.doc)
        .def("rhoJ", &Class::rhoJ, cls_doc.rhoJ.doc)
        .def("set_A", &Class::set_A, py::arg("A"), cls_doc.set_A.doc)
        .def("set_I1", &Class::set_I1, py::arg("I1"), cls_doc.set_I1.doc)
        .def("set_I2", &Class::set_I2, py::arg("I2"), cls_doc.set_I2.doc);
    DefCopyAndDeepCopy(&cls);
  }

  {
    using Class = DerUndeformedState<T>;
    constexpr auto& cls_doc = doc.DerUndeformedState;
    auto cls = DefineTemplateClassWithDefault<Class>(
        m, "DerUndeformedState", param, cls_doc.doc);
    cls  // BR
        .def_static("ZeroCurvatureAndTwist", &Class::ZeroCurvatureAndTwist,
            py::arg("has_closed_ends"), py::arg("edge_length"),
            cls_doc.ZeroCurvatureAndTwist.doc)
        .def("has_closed_ends", &Class::has_closed_ends,
            cls_doc.has_closed_ends.doc)
        .def("num_nodes", &Class::num_nodes, cls_doc.num_nodes.doc)
        .def("num_edges", &Class::num_edges, cls_doc.num_edges.doc)
        .def("num_internal_nodes", &Class::num_internal_nodes,
            cls_doc.num_internal_nodes.doc)
        .def("num_dofs", &Class::num_dofs, cls_doc.num_dofs.doc)
        .def("get_edge_length", &Class::get_edge_length,
            cls_doc.get_edge_length.doc)
        .def("get_voronoi_length", &Class::get_voronoi_length,
            cls_doc.get_voronoi_length.doc)
        .def("get_curvature_kappa1", &Class::get_curvature_kappa1,
            cls_doc.get_curvature_kappa1.doc)
        .def("get_curvature_kappa2", &Class::get_curvature_kappa2,
            cls_doc.get_curvature_kappa2.doc)
        .def("get_twist", &Class::get_twist, cls_doc.get_twist.doc)
        .def("set_edge_length", &Class::set_edge_length, py::arg("edge_length"),
            cls_doc.set_edge_length.doc)
        .def("set_curvature_kappa", &Class::set_curvature_kappa,
            py::arg("kappa1"), py::arg("kappa2"),
            cls_doc.set_curvature_kappa.doc)
        .def("set_curvature_angle", &Class::set_curvature_angle,
            py::arg("angle1"), py::arg("angle2"),
            cls_doc.set_curvature_angle.doc)
        .def("set_twist", &Class::set_twist, py::arg("twist"),
            cls_doc.set_twist.doc);
    DefCopyAndDeepCopy(&cls);
  }

  {
    using Class = DerModel<T>;
    constexpr auto& cls_doc = doc.DerModel;
    auto cls = DefineTemplateClassWithDefault<Class>(
        m, "DerModel", param, cls_doc.doc);
    cls  // BR
        .def("has_closed_ends", &Class::has_closed_ends,
            cls_doc.has_closed_ends.doc)
        .def("num_nodes", &Class::num_nodes, cls_doc.num_nodes.doc)
        .def("num_edges", &Class::num_edges, cls_doc.num_edges.doc)
        .def("num_dofs", &Class::num_dofs, cls_doc.num_dofs.doc)
        .def("structural_property", &Class::structural_property,
            cls_doc.structural_property.doc)
        .def("mutable_structural_property", &Class::mutable_structural_property,
            py_rvp::reference_internal, cls_doc.mutable_structural_property.doc)
        .def("mutable_undeformed_state", &Class::mutable_undeformed_state,
            py_rvp::reference_internal, cls_doc.mutable_undeformed_state.doc);
    DefClone(&cls);
  }
}
}  // namespace

PYBIND11_MODULE(der, m) {
  PYDRAKE_PREVENT_PYTHON3_MODULE_REIMPORT(m);
  m.doc() = "Bindings for multibody der.";

  type_visit([m](auto dummy) { DoScalarDependentDefinitions(m, dummy); },
      CommonScalarPack{});
}

}  // namespace pydrake
}  // namespace drake
