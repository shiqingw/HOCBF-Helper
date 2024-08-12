#ifndef PROBLEM_3D_COLLECTION_MOVING_OBSTACLE_HPP
#define PROBLEM_3D_COLLECTION_MOVING_OBSTACLE_HPP
#include <memory>
#include <tuple>
#include <chrono>
#include <limits>
#include <xtensor/xtensor.hpp>
#include <xtensor-python/pyarray.hpp>
#include <xtensor/xnoalias.hpp>

#include "threadPool.hpp"
#include "problem3d.hpp"
#include "utils.hpp"


class Problem3dCollectionMovingObstacle {
public:
    int n_problems = 0;
    int n_threads;
    ThreadPool thread_pool;
    std::vector<std::shared_ptr<Problem3d>> problems;
    std::vector<int> rob_frame_ids;
    std::vector<int> obs_frame_ids;

    Problem3dCollectionMovingObstacle(size_t num_threads);

    ~Problem3dCollectionMovingObstacle();

    void addProblem(std::shared_ptr<Problem3d> problem, int rob_frame_id, int obs_frame_id);

    void waitAll();

    void stopAll();

    std::tuple<xt::xtensor<double, 1>, xt::xtensor<double, 2>, xt::xtensor<double, 3>> solveGradientAndHessian(
        const xt::xtensor<double, 2>& all_d_rob, const xt::xtensor<double, 2>& all_q_rob,
        const xt::xtensor<double, 2>& all_d_obs, const xt::xtensor<double, 2>& all_q_obs);

    std::tuple<xt::xtensor<double, 1>, xt::xtensor<double, 2>, xt::xtensor<double, 3>, xt::xtensor<double, 1>, 
        xt::xtensor<double, 2>, xt::xtensor<double, 1>, xt::xtensor<double, 1>> getCBFConstraints(const xt::xtensor<double, 1>& dq,
        const xt::xtensor<double, 2>& all_postion, const xt::xtensor<double, 2>& all_quat, const xt::xtensor<double, 3>& all_Jacobian, 
        const xt::xtensor<double, 2>& all_dJdq, double alpha0, double gamma1, double gamma2, double compensation);

    // std::tuple<xt::xtensor<double, 1>, xt::xtensor<double, 2>, xt::xtensor<double, 3>, xt::xtensor<double, 1>, 
    //     xt::xtensor<double, 1>, xt::xtensor<double, 2>> getSmoothMinCBFConstraints(const xt::xtensor<double, 1>& dq,
    //     const xt::xtensor<double, 2>& all_postion, const xt::xtensor<double, 2>& all_quat, const xt::xtensor<double, 3>& all_Jacobian, 
    //     const xt::xtensor<double, 2>& all_dJdq, double alpha0);

};

#endif // PROBLEM_3D_COLLECTION_MOVING_OBSTACLE_HPP