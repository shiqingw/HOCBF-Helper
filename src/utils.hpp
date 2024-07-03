#ifndef UTILS_HPP
#define UTILS_HPP
#include <xtensor/xarray.hpp>

xt::xarray<double> getRotMatrixFromQuat(const xt::xarray<double>& q);

xt::xarray<double> getQMatrixFromQuat(const xt::xarray<double>& q);

xt::xarray<double> getdQMatrixFromdQuat(const xt::xarray<double>& dq);

#endif