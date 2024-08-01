#include <numeric>
#include <memory> 
#include <xtensor.hpp>
#include <pybind11/stl.h>
#include <pybind11/pybind11.h>
#define FORCE_IMPORT_ARRAY
#include <xtensor-python/pyarray.hpp>

#include "problem3d.hpp"
#include "ellipsoidAndLogSumExp3dPrb.hpp"
#include "ellipsoidAndHyperplane3dPrb.hpp"
#include "problem3dCollection.hpp"

#include "problem2d.hpp"
#include "ellipsoidAndLogSumExp2dPrb.hpp"
#include "ellipsoidAndHyperplane2dPrb.hpp"
#include "problem2dCollection.hpp"

namespace py = pybind11;

PYBIND11_MODULE(HOCBFHelperPy, m) {
    xt::import_numpy();
    m.doc() = "HOCBFHelperPy";

    py::class_<Problem3dCollection, std::shared_ptr<Problem3dCollection>>(m, "Problem3dCollection")
        .def(py::init<size_t>())
        .def_readonly("n_problems", &Problem3dCollection::n_problems)
        .def_readonly("n_threads", &Problem3dCollection::n_threads)
        .def_readonly("frame_ids", &Problem3dCollection::frame_ids)
        .def("addProblem", &Problem3dCollection::addProblem)
        .def("waitAll", &Problem3dCollection::waitAll)
        .def("stopAll", &Problem3dCollection::stopAll)
        .def("solveGradientAndHessian", &Problem3dCollection::solveGradientAndHessian)
        .def("getCBFConstraints", &Problem3dCollection::getCBFConstraints)
        .def("getSmoothMinCBFConstraints", &Problem3dCollection::getSmoothMinCBFConstraints);

    py::class_<Problem3d, std::shared_ptr<Problem3d>>(m, "Problem3d");

    py::class_<EllipsoidAndLogSumExp3dPrb, Problem3d, std::shared_ptr<EllipsoidAndLogSumExp3dPrb>>(m, "EllipsoidAndLogSumExp3dPrb")
        .def(py::init<std::shared_ptr<Ellipsoid3d>, std::shared_ptr<LogSumExp3d>, xt::xtensor<double, 2>>())
        .def("solveSCSPrb", &EllipsoidAndLogSumExp3dPrb::solveSCSPrb)
        .def("solve", &EllipsoidAndLogSumExp3dPrb::solve)
        .def_readonly("dim_p", &EllipsoidAndLogSumExp3dPrb::dim_p_)
        .def_readonly("n_exp_cone", &EllipsoidAndLogSumExp3dPrb::n_exp_cone_)
        .def_readonly("A_scs", &EllipsoidAndLogSumExp3dPrb::A_scs_)
        .def_readonly("b_scs", &EllipsoidAndLogSumExp3dPrb::b_scs_)
        .def_readonly("p_sol", &EllipsoidAndLogSumExp3dPrb::p_sol_)
        .def(py::pickle(
            [](const EllipsoidAndLogSumExp3dPrb &p) {
                return py::make_tuple(p.SF_rob_, p.SF_obs_, p.obs_characteristic_points_);
            },
            [](py::tuple t) {
                if (t.size() != 3)
                    throw std::runtime_error("Invalid state!");
                return std::make_shared<EllipsoidAndLogSumExp3dPrb>(t[0].cast<std::shared_ptr<Ellipsoid3d>>(), t[1].cast<std::shared_ptr<LogSumExp3d>>(),
                    t[2].cast<xt::xtensor<double, 2>>());
            }
        ));
    
    py::class_<EllipsoidAndHyperplane3dPrb, Problem3d, std::shared_ptr<EllipsoidAndHyperplane3dPrb>>(m, "EllipsoidAndHyperplane3dPrb")
        .def(py::init<std::shared_ptr<Ellipsoid3d>, std::shared_ptr<Hyperplane3d>>())
        .def("solve", &EllipsoidAndHyperplane3dPrb::solve)
        .def_readonly("SF_rob", &EllipsoidAndHyperplane3dPrb::SF_rob_)
        .def_readonly("SF_obs", &EllipsoidAndHyperplane3dPrb::SF_obs_)
        .def_readonly("p_sol", &EllipsoidAndHyperplane3dPrb::p_sol_)
        .def(py::pickle(
            [](const EllipsoidAndHyperplane3dPrb &p) {
                return py::make_tuple(p.SF_rob_, p.SF_obs_);
            },
            [](py::tuple t) {
                if (t.size() != 2)
                    throw std::runtime_error("Invalid state!");
                return std::make_shared<EllipsoidAndHyperplane3dPrb>(t[0].cast<std::shared_ptr<Ellipsoid3d>>(), t[1].cast<std::shared_ptr<Hyperplane3d>>());
            }
        ));

    py::class_<Problem2d, std::shared_ptr<Problem2d>>(m, "Problem2d");

    py::class_<Problem2dCollection, std::shared_ptr<Problem2dCollection>>(m, "Problem2dCollection")
        .def(py::init<size_t>())
        .def_readonly("n_problems", &Problem2dCollection::n_problems)
        .def_readonly("n_threads", &Problem2dCollection::n_threads)
        .def_readonly("frame_ids", &Problem2dCollection::frame_ids)
        .def("addProblem", &Problem2dCollection::addProblem)
        .def("waitAll", &Problem2dCollection::waitAll)
        .def("stopAll", &Problem2dCollection::stopAll)
        .def("solveGradientAndHessian", &Problem2dCollection::solveGradientAndHessian)
        .def("getCBFConstraints", &Problem2dCollection::getCBFConstraints)
        .def("getCBFConstraintsFixedOrientation", &Problem2dCollection::getCBFConstraintsFixedOrientation)
        .def("getSmoothMinCBFConstraints", &Problem2dCollection::getSmoothMinCBFConstraints)
        .def("getSmoothMinCBFConstraintsFixedOrientation", &Problem2dCollection::getSmoothMinCBFConstraintsFixedOrientation);

    py::class_<EllipsoidAndLogSumExp2dPrb, Problem2d, std::shared_ptr<EllipsoidAndLogSumExp2dPrb>>(m, "EllipsoidAndLogSumExp2dPrb")
        .def(py::init<std::shared_ptr<Ellipsoid2d>, std::shared_ptr<LogSumExp2d>, xt::xtensor<double, 2>>())
        .def("solveSCSPrb", &EllipsoidAndLogSumExp2dPrb::solveSCSPrb)
        .def("solve", &EllipsoidAndLogSumExp2dPrb::solve)
        .def_readonly("dim_p", &EllipsoidAndLogSumExp2dPrb::dim_p_)
        .def_readonly("n_exp_cone", &EllipsoidAndLogSumExp2dPrb::n_exp_cone_)
        .def_readonly("A_scs", &EllipsoidAndLogSumExp2dPrb::A_scs_)
        .def_readonly("b_scs", &EllipsoidAndLogSumExp2dPrb::b_scs_)
        .def_readonly("p_sol", &EllipsoidAndLogSumExp2dPrb::p_sol_)
        .def(py::pickle(
            [](const EllipsoidAndLogSumExp2dPrb &p) {
                return py::make_tuple(p.SF_rob_, p.SF_obs_, p.obs_characteristic_points_);
            },
            [](py::tuple t) {
                if (t.size() != 3)
                    throw std::runtime_error("Invalid state!");
                return std::make_shared<EllipsoidAndLogSumExp2dPrb>(t[0].cast<std::shared_ptr<Ellipsoid2d>>(), t[1].cast<std::shared_ptr<LogSumExp2d>>(),
                    t[2].cast<xt::xtensor<double, 2>>());
            }
        ));

    py::class_<EllipsoidAndHyperplane2dPrb, Problem2d, std::shared_ptr<EllipsoidAndHyperplane2dPrb>>(m, "EllipsoidAndHyperplane2dPrb")
        .def(py::init<std::shared_ptr<Ellipsoid2d>, std::shared_ptr<Hyperplane2d>>())
        .def("solve", &EllipsoidAndHyperplane2dPrb::solve)
        .def_readonly("SF_rob", &EllipsoidAndHyperplane2dPrb::SF_rob_)
        .def_readonly("SF_obs", &EllipsoidAndHyperplane2dPrb::SF_obs_)
        .def_readonly("p_sol", &EllipsoidAndHyperplane2dPrb::p_sol_)
        .def(py::pickle(
            [](const EllipsoidAndHyperplane2dPrb &p) {
                return py::make_tuple(p.SF_rob_, p.SF_obs_);
            },
            [](py::tuple t) {
                if (t.size() != 2)
                    throw std::runtime_error("Invalid state!");
                return std::make_shared<EllipsoidAndHyperplane2dPrb>(t[0].cast<std::shared_ptr<Ellipsoid2d>>(), t[1].cast<std::shared_ptr<Hyperplane2d>>());
            }
        ));

}