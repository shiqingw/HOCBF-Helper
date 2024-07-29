#ifndef PROBLEM_2D_COLLECTION_HPP
#define PROBLEM_2D_COLLECTION_HPP
#include <memory>
#include <tuple>
#include <chrono>
#include <limits>
#include <xtensor/xtensor.hpp>
#include <xtensor-python/pyarray.hpp>
#include <xtensor/xnoalias.hpp>

#include "threadPool.hpp"
#include "problem2d.hpp"
#include "utils.hpp"


class Problem2dCollection {
public:
    int n_problems = 0;
    int n_threads;
    ThreadPool thread_pool;
    std::vector<std::shared_ptr<Problem2d>> problems;
    std::vector<int> frame_ids;

    Problem2dCollection(size_t num_threads);

    ~Problem2dCollection();

    void addProblem(std::shared_ptr<Problem2d> problem, int frame_id);

    void waitAll();

    void stopAll();

    std::tuple<xt::xtensor<double, 1>, xt::xtensor<double, 2>, xt::xtensor<double, 3>> solveGradientAndHessian(
        const xt::xtensor<double, 2>& all_d, const xt::xtensor<double, 1>& all_theta);

};

#endif