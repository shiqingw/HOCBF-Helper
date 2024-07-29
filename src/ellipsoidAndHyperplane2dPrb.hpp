#ifndef ELLIPSOID_AND_HYPERPLANE_2D_PRB_HPP
#define ELLIPSOID_AND_HYPERPLANE_2D_PRB_HPP
#include <memory>
#include <tuple>
#include <chrono>
#include <limits>
#include <xtensor/xtensor.hpp>
#include <xtensor/xio.hpp>
#include <scalingFunctionsHelper/ellipsoid2d.hpp>
#include <scalingFunctionsHelper/hyperplane2d.hpp>
#include <scalingFunctionsHelper/diffOptHelper.hpp>

#include "problem2d.hpp"

class EllipsoidAndHyperplane2dPrb: public Problem2d {
public:
    std::shared_ptr<Ellipsoid2d> SF_rob_;
    std::shared_ptr<Hyperplane2d> SF_obs_;
    xt::xtensor<double, 2> p_sol_;

    EllipsoidAndHyperplane2dPrb(std::shared_ptr<Ellipsoid2d> SF_rob, std::shared_ptr<Hyperplane2d> SF_obs) :
        SF_rob_(SF_rob), SF_obs_(SF_obs) { };

    ~EllipsoidAndHyperplane2dPrb() = default;

    std::tuple<double, xt::xtensor<double, 1>, xt::xtensor<double, 2>> solve(const xt::xtensor<double, 1>& d, double theta) override;

};

#endif // ELLIPSOID_AND_HYPERPLANE_2D_PRB_HPP