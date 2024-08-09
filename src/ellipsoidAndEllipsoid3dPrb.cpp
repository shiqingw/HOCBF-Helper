#include "ellipsoidAndEllipsoid3dPrb.hpp"

std::tuple<double, xt::xtensor<double, 1>, xt::xtensor<double, 2>> EllipsoidAndEllipsoid3dPrb::solve(
    const xt::xtensor<double, 1>& d, const xt::xtensor<double, 1>& q) {
    
    xt::xtensor<double, 1> d2 = xt::zeros<double>({3});
    xt::xtensor<double, 1> q2 = xt::zeros<double>({4});
    q2(3) = 1;

    p_sol_ = rimonMethod3d(SF_rob_, d, q, SF_obs_, d2, q2);

    double alpha;
    xt::xtensor<double, 1> alpha_dx;
    xt::xtensor<double, 2> alpha_dxdx;

    std::tie(alpha, alpha_dx, alpha_dxdx) = getGradientAndHessian3d(p_sol_, SF_rob_, d, q, SF_obs_, d2, q2);

    return std::make_tuple(alpha, alpha_dx, alpha_dxdx);
}

std::tuple<double, xt::xtensor<double, 1>, xt::xtensor<double, 2>> EllipsoidAndEllipsoid3dPrb::solveMovingObstacle(
    const xt::xtensor<double, 1>& d1, const xt::xtensor<double, 1>& q1, const xt::xtensor<double, 1>& d2, 
    const xt::xtensor<double, 1>& q2) {

    p_sol_ = rimonMethod3d(SF_rob_, d1, q1, SF_obs_, d2, q2);

    double alpha;
    xt::xtensor<double, 1> alpha_dx;
    xt::xtensor<double, 2> alpha_dxdx;

    std::tie(alpha, alpha_dx, alpha_dxdx) = getGradientAndHessian3d(p_sol_, SF_rob_, d1, q1, SF_obs_, d2, q2);

    return std::make_tuple(alpha, alpha_dx, alpha_dxdx);
}