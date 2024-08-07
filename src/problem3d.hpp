#ifndef PROBLEM_3D_HPP
#define PROBLEM_3D_HPP
#include <memory>
#include <xtensor/xarray.hpp>

#include <scalingFunctionsHelper/scalingFunction3d.hpp>

class Problem3d {
public:

    Problem3d();

    ~Problem3d()=default;

    virtual std::tuple<double, xt::xtensor<double, 1>, xt::xtensor<double, 2>> solve(const xt::xtensor<double, 1>& d, const xt::xtensor<double, 1>& q)=0;

};

#endif