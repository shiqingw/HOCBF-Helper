#ifndef PROBLEM_3D_HPP
#define PROBLEM_3D_HPP
#include <memory>
#include <xtensor/xarray.hpp>

#include <scalingFunctionsHelper/scalingFunction3d.hpp>

class Problem3d {
public:

    Problem3d();

    ~Problem3d()=default;

    virtual std::tuple<double, xt::xarray<double>, xt::xarray<double>> solve(const xt::xarray<double>& d, const xt::xarray<double>& q)=0;

};

#endif