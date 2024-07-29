#include "ellipsoidAndHyperplane3dPrb.hpp"

std::tuple<double, xt::xtensor<double, 1>, xt::xtensor<double, 2>> EllipsoidAndHyperplane3dPrb::solve(
    const xt::xtensor<double, 1>& d, const xt::xtensor<double, 1>& q) {

    xt::xtensor<double, 2> Q = SF_rob_->getWorldQuadraticCoefficient(q);
    xt::xtensor<double, 1> mu = SF_rob_->getWorldCenter(d, q);

    xt::xtensor<double, 1> a = SF_obs_->a;
    double b = SF_obs_->b;

    xt::xtensor<double, 2> Q_inv = xt::linalg::inv(Q);
    double tmp = std::max(0.0, xt::linalg::dot(a, mu)(0) + b) / xt::linalg::dot(a, xt::linalg::dot(Q_inv, a))(0);
    xt::xtensor<double, 1> p = mu - xt::linalg::dot(Q_inv, a) * tmp;

    p_sol_ = p;

    double alpha;
    xt::xtensor<double, 1> alpha_dx;
    xt::xtensor<double, 2> alpha_dxdx;
    xt::xtensor<double, 1> d2 = xt::zeros<double>({3});
    xt::xtensor<double, 1> q2 = xt::zeros<double>({4});
    q2(3) = 1;

    std::tie(alpha, alpha_dx, alpha_dxdx) = getGradientAndHessian3d(p, SF_rob_, d, q, SF_obs_, d2, q2);

    return std::make_tuple(alpha, alpha_dx, alpha_dxdx);
}
