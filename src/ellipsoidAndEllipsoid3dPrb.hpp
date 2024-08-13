#ifndef ELLIPSOID_AND_ELLIPSOID_3D_PRB_HPP
#define ELLIPSOID_AND_ELLIPSOID_3D_PRB_HPP
#include <memory>
#include <tuple>
#include <chrono>
#include <limits>
#include <xtensor/xtensor.hpp>
#include <xtensor/xio.hpp>
#include <scalingFunctionsHelper/ellipsoid3d.hpp>
#include <scalingFunctionsHelper/rimonMethod.hpp>
#include <scalingFunctionsHelper/diffOptHelper.hpp>

#include "problem3d.hpp"

class EllipsoidAndEllipsoid3dPrb: public Problem3d {
public:
    std::shared_ptr<Ellipsoid3d> SF_rob_;
    std::shared_ptr<Ellipsoid3d> SF_obs_;
    xt::xtensor<double, 1> p_sol_;

    EllipsoidAndEllipsoid3dPrb(std::shared_ptr<Ellipsoid3d> SF_rob, std::shared_ptr<Ellipsoid3d> SF_obs) :
        SF_rob_(SF_rob), SF_obs_(SF_obs) {
        p_sol_ = xt::zeros<double>({3});
    };

    ~EllipsoidAndEllipsoid3dPrb() = default;

    void validateProblem(int rob_frame_id, int obs_frame_id) override;

    std::tuple<double, xt::xtensor<double, 1>, xt::xtensor<double, 2>> solve(const xt::xtensor<double, 1>& d, 
        const xt::xtensor<double, 1>& q) override;

    std::tuple<double, xt::xtensor<double, 1>, xt::xtensor<double, 2>> solveMovingObstacle(
        const xt::xtensor<double, 1>& d1, const xt::xtensor<double, 1>& q1, const xt::xtensor<double, 1>& d2,
        const xt::xtensor<double, 1>& q2) override;

};


#endif // ELLIPSOID_AND_ELLIPSOID_3D_PRB_HPP