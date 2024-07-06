#ifndef PROBLEM_3D_COLLECTION_HPP
#define PROBLEM_3D_COLLECTION_HPP
#include <memory>
#include <tuple>
#include <chrono>
#include <limits>
#include <xtensor/xarray.hpp>
#include <xtensor-python/pyarray.hpp>

#include "threadPool.hpp"
#include "problem3d.hpp"
#include "utils.hpp"


class Problem3dCollection {
public:
    int n_problems = 0;
    int n_threads;
    ThreadPool thread_pool;
    std::vector<std::shared_ptr<Problem3d>> problems;
    std::vector<int> frame_ids;

    Problem3dCollection(size_t num_threads);

    ~Problem3dCollection();

    void addProblem(std::shared_ptr<Problem3d> problem, int frame_id);

    void waitAll();

    std::tuple<xt::xarray<double>, xt::xarray<double>, xt::xarray<double>> solveGradientAndHessian(
        const xt::xarray<double>& all_d, const xt::xarray<double>& all_q);

    std::tuple<xt::xarray<double>, xt::xarray<double>, xt::xarray<double>, xt::xarray<double>, 
        xt::xarray<double>, xt::xarray<double>, xt::xarray<double>> getCBFConstraints(const xt::xarray<double>& dq,
        const xt::xarray<double>& all_postion, const xt::xarray<double>& all_quat, const xt::xarray<double>& all_Jacobian, 
        const xt::xarray<double>& all_dJdq, double alpha0, double gamma1, double gamma2, double compensation);

    std::tuple<xt::xarray<double>, xt::xarray<double>, xt::xarray<double>, xt::xarray<double>, 
        xt::xarray<double>, xt::xarray<double>> getSmoothMinCBFConstraints(const xt::xarray<double>& dq,
        const xt::xarray<double>& all_postion, const xt::xarray<double>& all_quat, const xt::xarray<double>& all_Jacobian, 
        const xt::xarray<double>& all_dJdq, double alpha0);

};

#endif