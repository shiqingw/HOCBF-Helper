#include <numeric>
#include <memory> 
#include <xtensor.hpp>
#include <pybind11/stl.h>
#include <pybind11/pybind11.h>
#define FORCE_IMPORT_ARRAY
#include <xtensor-python/pyarray.hpp>

#include "problem3d.hpp"
#include "elliposoidAndLogSumExp3dPrb.hpp"
#include "problem3dCollection.hpp"

namespace py = pybind11;

PYBIND11_MODULE(HOCBFHelperPy, m) {
    xt::import_numpy();
    m.doc() = "HOCBFHelperPy";

    py::class_<Problem3d, std::shared_ptr<Problem3d>>(m, "Problem3d");

    py::class_<ElliposoidAndLogSumExp3dPrb, Problem3d, std::shared_ptr<ElliposoidAndLogSumExp3dPrb>>(m, "ElliposoidAndLogSumExp3dPrb")
        .def(py::init<std::shared_ptr<Ellipsoid3d>, std::shared_ptr<LogSumExp3d>>())
        .def("solve_scs_prb", &ElliposoidAndLogSumExp3dPrb::solve_scs_prb)
        .def("solve", &ElliposoidAndLogSumExp3dPrb::solve)
        .def_readonly("dim_p", &ElliposoidAndLogSumExp3dPrb::dim_p_)
        .def_readonly("n_exp_cone", &ElliposoidAndLogSumExp3dPrb::n_exp_cone_)
        .def_readonly("A_scs", &ElliposoidAndLogSumExp3dPrb::A_scs_)
        .def_readonly("b_scs", &ElliposoidAndLogSumExp3dPrb::b_scs_)
        .def_readonly("p_sol", &ElliposoidAndLogSumExp3dPrb::p_sol_)
        .def(py::pickle(
            [](const ElliposoidAndLogSumExp3dPrb &p) {
                return py::make_tuple(p.SF_rob_, p.SF_obs_);
            },
            [](py::tuple t) {
                if (t.size() != 2)
                    throw std::runtime_error("Invalid state!");
                return std::make_shared<ElliposoidAndLogSumExp3dPrb>(t[0].cast<std::shared_ptr<Ellipsoid3d>>(), t[1].cast<std::shared_ptr<LogSumExp3d>>());
            }
        ));
    
    py::class_<Problem3dCollection, std::shared_ptr<Problem3dCollection>>(m, "Problem3dCollection")
        .def(py::init<size_t>())
        .def("add_problem", &Problem3dCollection::add_problem)
        .def("solve_all", &Problem3dCollection::solve_all)
        .def("wait_all", &Problem3dCollection::wait_all);
}