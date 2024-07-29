#include "ellipsoidAndHyperplane2dPrb.hpp"

std::tuple<double, xt::xtensor<double, 1>, xt::xtensor<double, 2>> EllipsoidAndHyperplane2dPrb::solve(
    const xt::xtensor<double, 1>& d, double theta) {

    xt::xtensor<double, 2> Q = SF_rob_->getWorldQuadraticCoefficient(theta);
    xt::xtensor<double, 1> mu = SF_rob_->getWorldCenter(d, theta);

    xt::xtensor<double, 1> a = SF_obs_->a;
    double b = SF_obs_->b;

    xt::xtensor<double, 2> Q_inv = xt::linalg::inv(Q);
    double tmp = std::max(0.0, xt::linalg::dot(a, mu)(0) + b) / xt::linalg::dot(a, xt::linalg::dot(Q_inv, a))(0);
    xt::xtensor<double, 1> p = mu - xt::linalg::dot(Q_inv, a) * tmp;

    p_sol_ = p;

    double alpha;
    xt::xtensor<double, 1> alpha_dx;
    xt::xtensor<double, 2> alpha_dxdx;
    xt::xtensor<double, 1> d2 = xt::zeros<double>({2});
    double theta2 = 0.0;

    std::tie(alpha, alpha_dx, alpha_dxdx) = getGradientAndHessian2d(p, SF_rob_, d, theta, SF_obs_, d2, theta2);

    return std::make_tuple(alpha, alpha_dx, alpha_dxdx);
}
