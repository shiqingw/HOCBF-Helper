#ifndef UTILS_HPP
#define UTILS_HPP
#include <xtensor/xtensor.hpp>

xt::xtensor<double, 2> getRotMatrixFromQuat(const xt::xtensor<double, 1>& q);

xt::xtensor<double, 2> getQMatrixFromQuat(const xt::xtensor<double, 1>& q);

xt::xtensor<double, 2> getdQMatrixFromdQuat(const xt::xtensor<double, 1>& dq);

#endif