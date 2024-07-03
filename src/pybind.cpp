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
        .def(py::init<std::shared_ptr<Ellipsoid3d>, std::shared_ptr<LogSumExp3d>, xt::xarray<double>>())
        .def("solveSCSPrb", &ElliposoidAndLogSumExp3dPrb::solveSCSPrb)
        .def("solve", &ElliposoidAndLogSumExp3dPrb::solve)
        .def_readonly("dim_p", &ElliposoidAndLogSumExp3dPrb::dim_p_)
        .def_readonly("n_exp_cone", &ElliposoidAndLogSumExp3dPrb::n_exp_cone_)
        .def_readonly("A_scs", &ElliposoidAndLogSumExp3dPrb::A_scs_)
        .def_readonly("b_scs", &ElliposoidAndLogSumExp3dPrb::b_scs_)
        .def_readonly("p_sol", &ElliposoidAndLogSumExp3dPrb::p_sol_)
        .def(py::pickle(
            [](const ElliposoidAndLogSumExp3dPrb &p) {
                return py::make_tuple(p.SF_rob_, p.SF_obs_, p.obs_characteristic_points_);
            },
            [](py::tuple t) {
                if (t.size() != 3)
                    throw std::runtime_error("Invalid state!");
                return std::make_shared<ElliposoidAndLogSumExp3dPrb>(t[0].cast<std::shared_ptr<Ellipsoid3d>>(), t[1].cast<std::shared_ptr<LogSumExp3d>>(),
                    t[2].cast<xt::xarray<double>>());
            }
        ));
    
    py::class_<Problem3dCollection, std::shared_ptr<Problem3dCollection>>(m, "Problem3dCollection")
        .def(py::init<size_t>())
        .def_readonly("n_problems", &Problem3dCollection::n_problems)
        .def_readonly("n_threads", &Problem3dCollection::n_threads)
        .def_readonly("frame_ids", &Problem3dCollection::frame_ids)
        .def("addProblem", &Problem3dCollection::addProblem)
        .def("solveGradientAndHessian", &Problem3dCollection::solveGradientAndHessian)
        .def("getCBFConstraints", &Problem3dCollection::getCBFConstraints)
        .def("waitAll", &Problem3dCollection::waitAll);

}