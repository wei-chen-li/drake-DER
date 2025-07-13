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

  constexpr auto& doc = pydrake_doc.drake.multibody.der;

  {
    using Class = drake::multibody::der::internal::DerUndeformedState<T>;
    auto cls =
        DefineTemplateClassWithDefault<Class>(m, "DerUndeformedState", param);
    cls  // BR
        .def_static("ZeroCurvatureAndTwist", &Class::ZeroCurvatureAndTwist,
            py::arg("has_closed_ends"), py::arg("edge_length"))
        .def("get_edge_length", &Class::get_edge_length)
        .def("get_voronoi_length", &Class::get_voronoi_length)
        .def("get_curvature_kappa1", &Class::get_curvature_kappa1)
        .def("get_curvature_kappa2", &Class::get_curvature_kappa2)
        .def("get_twist", &Class::get_twist)
        .def("set_edge_length", &Class::set_edge_length, py::arg("edge_length"))
        .def("set_curvature_kappa", &Class::set_curvature_kappa,
            py::arg("kappa1"), py::arg("kappa2"))
        .def("set_twist", &Class::set_twist, py::arg("twist"));
    DefCopyAndDeepCopy(&cls);
  }

  {
    using Class = drake::multibody::der::DerModel<T>;
    constexpr auto& cls_doc = doc.DerModel;
    auto cls = DefineTemplateClassWithDefault<Class>(
        m, "DerModel", param, cls_doc.doc);
    cls  // BR
        .def("has_closed_ends", &Class::has_closed_ends,
            cls_doc.has_closed_ends.doc)
        .def("num_nodes", &Class::num_nodes, cls_doc.num_nodes.doc)
        .def("num_edges", &Class::num_edges, cls_doc.num_edges.doc)
        .def("num_dofs", &Class::num_dofs, cls_doc.num_dofs.doc)
        .def("mutable_undeformed_state", &Class::mutable_undeformed_state,
            cls_doc.mutable_undeformed_state.doc);
    DefClone(&cls);
  }
}
}  // namespace

PYBIND11_MODULE(der, m) {
  PYDRAKE_PREVENT_PYTHON3_MODULE_REIMPORT(m);
  m.doc() = "Bindings for multibody der.";

  type_visit([m](auto dummy) { DoScalarDependentDefinitions(m, dummy); },
      NonSymbolicScalarPack{});
}

}  // namespace pydrake
}  // namespace drake
