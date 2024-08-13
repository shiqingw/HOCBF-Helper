#ifndef PROBLEM_3D_HPP
#define PROBLEM_3D_HPP
#include <memory>
#include <xtensor/xarray.hpp>

#include <scalingFunctionsHelper/scalingFunction3d.hpp>

class Problem3d {
public:
    std::shared_ptr<ScalingFunction3d> SF_rob_;
    std::shared_ptr<ScalingFunction3d> SF_obs_;

    Problem3d();

    ~Problem3d()=default;

    virtual std::tuple<double, xt::xtensor<double, 1>, xt::xtensor<double, 2>> solve(const xt::xtensor<double, 1>& d, 
        const xt::xtensor<double, 1>& q) = 0;

    virtual std::tuple<double, xt::xtensor<double, 1>, xt::xtensor<double, 2>> solveMovingObstacle(
        const xt::xtensor<double, 1>& d1, const xt::xtensor<double, 1>& q1, const xt::xtensor<double, 1>& d2,
        const xt::xtensor<double, 1>& q2) = 0;
    
    virtual void validateProblem(int rob_frame_id, int obs_frame_id) = 0;

};

#endif