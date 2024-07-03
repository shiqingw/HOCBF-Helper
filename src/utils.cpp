#include "utils.hpp"

xt::xarray<double> getRotMatrixFromQuat(const xt::xarray<double>& q){
    double qx = q(0), qy = q(1), qz = q(2), qw = q(3);
    xt::xarray<double> R = xt::zeros<double>({3, 3});

    R(0,0) = 2*std::pow(qw, 2) + 2*std::pow(qx, 2) - 1;
    R(0,1) = -2*qw*qz + 2*qx*qy;
    R(0,2) = 2*qw*qy + 2*qx*qz;
    R(1,0) = 2*qw*qz + 2*qx*qy;
    R(1,1) = 2*std::pow(qw, 2) + 2*std::pow(qy, 2) - 1;
    R(1,2) = -2*qw*qx + 2*qy*qz;
    R(2,0) = -2*qw*qy + 2*qx*qz;
    R(2,1) = 2*qw*qx + 2*qy*qz;
    R(2,2) = 2*std::pow(qw, 2) + 2*std::pow(qz, 2) - 1;

    return R;
};

xt::xarray<double> getQMatrixFromQuat(const xt::xarray<double>& q){

    double qx = q(0), qy = q(1), qz = q(2), qw = q(3);
    xt::xarray<double> Q {{qw, qz, -qy},
                            {-qz, qw, qx},
                            {qy, -qx, qw},
                            {-qx, -qy, -qz}};
    return Q;
};

xt::xarray<double> getdQMatrixFromdQuat(const xt::xarray<double>& dq){

    double dqx = dq(0), dqy = dq(1), dqz = dq(2), dqw = dq(3);
    xt::xarray<double> dQ {{dqw, dqz, -dqy},
                            {-dqz, dqw, dqx},
                            {dqy, -dqx, dqw},
                            {-dqx, -dqy, -dqz}};
    return dQ;
};

