#ifndef ELLIPSOID_AND_LOG_SUM_EXP_3D_PRB_HPP
#define ELLIPSOID_AND_LOG_SUM_EXP_3D_PRB_HPP
#include <memory>
#include <tuple>
#include <chrono>
#include <limits>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>
#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <scs/scs.h>
#include <scalingFunctionsHelper/ellipsoid3d.hpp>
#include <scalingFunctionsHelper/logSumExp3d.hpp>
#include <scalingFunctionsHelper/diffOptHelper.hpp>

#include "problem3d.hpp"

class ElliposoidAndLogSumExp3dPrb: public Problem3d{
public:
    std::shared_ptr<Ellipsoid3d> SF_rob_;
    std::shared_ptr<LogSumExp3d> SF_obs_;
    xt::xarray<double> obs_characteristic_points_;

    int dim_p_ = 3;
    int n_exp_cone_;
    int n_;
    int m_;
    Eigen::SparseMatrix<double> A_scs_;
    Eigen::VectorXd b_scs_;
    xt::xarray<double> p_sol_;

    double* warm_x_;
    double* warm_y_;
    double* warm_s_;

    ScsSolution* scs_sol_;
    ScsSettings* scs_stgs_;
    ScsInfo* scs_info_;
    ScsCone* scs_cone_;

    ElliposoidAndLogSumExp3dPrb(std::shared_ptr<Ellipsoid3d> SF_rob, std::shared_ptr<LogSumExp3d> SF_obs, 
        const xt::xarray<double>& obs_characteristic_points);

    ~ElliposoidAndLogSumExp3dPrb();

    std::tuple<int, xt::xarray<double>> solveSCSPrb(const xt::xarray<double>& d, const xt::xarray<double>& q);

    std::tuple<double, xt::xarray<double>, xt::xarray<double>> solve(const xt::xarray<double>& d, const xt::xarray<double>& q) override;

};


#endif // ELLIPSOID_AND_LOG_SUM_EXP_3D_PRB_HPP