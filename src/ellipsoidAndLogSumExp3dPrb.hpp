#ifndef ELLIPSOID_AND_LOG_SUM_EXP_3D_PRB_HPP
#define ELLIPSOID_AND_LOG_SUM_EXP_3D_PRB_HPP
#include <memory>
#include <tuple>
#include <chrono>
#include <limits>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>
#include <xtensor/xtensor.hpp>
#include <xtensor/xio.hpp>
#include <scs/scs.h>
#include <scalingFunctionsHelper/ellipsoid3d.hpp>
#include <scalingFunctionsHelper/logSumExp3d.hpp>
#include <scalingFunctionsHelper/diffOptHelper.hpp>

#include "problem3d.hpp"

class EllipsoidAndLogSumExp3dPrb: public Problem3d{
public:
    std::shared_ptr<Ellipsoid3d> SF_rob_;
    std::shared_ptr<LogSumExp3d> SF_obs_;
    xt::xtensor<double, 2> obs_characteristic_points_;

    int dim_p_ = 3;
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

    EllipsoidAndLogSumExp3dPrb(std::shared_ptr<Ellipsoid3d> SF_rob, std::shared_ptr<LogSumExp3d> SF_obs, 
        const xt::xtensor<double, 2>& obs_characteristic_points);

    ~EllipsoidAndLogSumExp3dPrb();

    std::tuple<int, xt::xtensor<double, 1>> solveSCSPrb(const xt::xtensor<double, 1>& d, const xt::xtensor<double, 1>& q);

    std::tuple<double, xt::xtensor<double, 1>, xt::xtensor<double, 2>> solve(const xt::xtensor<double, 1>& d, const xt::xtensor<double, 1>& q) override;

};


#endif // ELLIPSOID_AND_LOG_SUM_EXP_3D_PRB_HPP