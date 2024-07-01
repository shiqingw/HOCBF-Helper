#ifndef PROBLEM_3D_COLLECTION_HPP
#define PROBLEM_3D_COLLECTION_HPP
#include <memory>
#include <tuple>
#include <chrono>
#include <xtensor/xarray.hpp>

#include "threadPool.hpp"
#include "problem3d.hpp"


class Problem3dCollection {
public:
    int n_problems = 0;
    int n_threads;
    ThreadPool thread_pool;
    std::vector<std::shared_ptr<Problem3d>> problems;

    Problem3dCollection(size_t num_threads);

    ~Problem3dCollection();

    void add_problem(std::shared_ptr<Problem3d> problem);

    std::tuple<xt::xarray<double>, xt::xarray<double>, xt::xarray<double>> solve_all(const xt::xarray<double>& all_d, 
        const xt::xarray<double>& all_q);

    void wait_all();
};

#endif