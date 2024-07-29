#ifndef PROBLEM_2D_HPP
#define PROBLEM_2D_HPP
#include <memory>
#include <xtensor/xarray.hpp>

#include <scalingFunctionsHelper/scalingFunction2d.hpp>

class Problem2d {
public:

    Problem2d();

    ~Problem2d()=default;

    virtual std::tuple<double, xt::xtensor<double, 1>, xt::xtensor<double, 2>> solve(const xt::xtensor<double, 1>& d, double theta)=0;

};

#endif