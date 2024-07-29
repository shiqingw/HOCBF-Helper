#ifndef ELLIPSOID_AND_LOG_SUM_EXP_2D_PRB_HPP
#define ELLIPSOID_AND_LOG_SUM_EXP_2D_PRB_HPP
#include <memory>
#include <tuple>
#include <chrono>
#include <limits>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>
#include <xtensor/xtensor.hpp>
#include <xtensor/xio.hpp>
#include <scs/scs.h>
#include <scalingFunctionsHelper/ellipsoid2d.hpp>
#include <scalingFunctionsHelper/logSumExp2d.hpp>
#include <scalingFunctionsHelper/diffOptHelper.hpp>

#include "problem2d.hpp"

class EllipsoidAndLogSumExp2dPrb: public Problem2d{
public:
    std::shared_ptr<Ellipsoid2d> SF_rob_;
    std::shared_ptr<LogSumExp2d> SF_obs_;
    xt::xtensor<double, 2> obs_characteristic_points_;

    int dim_p_ = 2;
    int n_exp_cone_;
    int n_;
    int m_;
    Eigen::SparseMatrix<double> A_scs_;
    Eigen::VectorXd b_scs_;
    xt::xtensor<double, 1> p_sol_;

    double* warm_x_;
    double* warm_y_;
    double* warm_s_;

    ScsSolution* scs_sol_;
    ScsSettings* scs_stgs_;
    ScsInfo* scs_info_;
    ScsCone* scs_cone_;

    EllipsoidAndLogSumExp2dPrb(std::shared_ptr<Ellipsoid2d> SF_rob, std::shared_ptr<LogSumExp2d> SF_obs, 
        const xt::xtensor<double, 2>& obs_characteristic_points);

    ~EllipsoidAndLogSumExp2dPrb();

    std::tuple<int, xt::xtensor<double, 1>> solveSCSPrb(const xt::xtensor<double, 1>& d, double theta);

    std::tuple<double, xt::xtensor<double, 1>, xt::xtensor<double, 2>> solve(const xt::xtensor<double, 1>& d, double theta) override;

};


#endif // ELLIPSOID_AND_LOG_SUM_EXP_2D_PRB_HPP