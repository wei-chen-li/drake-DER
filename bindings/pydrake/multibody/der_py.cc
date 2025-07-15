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
      NonSymbolicScalarPack{});
}

}  // namespace pydrake
}  // namespace drake
