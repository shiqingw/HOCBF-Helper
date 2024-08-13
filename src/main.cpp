#include <iostream>
#include <memory>
#include <xtensor/xtensor.hpp>
#include <xtensor/xio.hpp>
#include <chrono>

#include "ellipsoidAndEllipsoid3dPrb.hpp"
#include "problem3dCollectionMovingObstacle.hpp"

int main() {
    xt::xtensor<double, 2> Q_1 = xt::eye<double>(3);
    xt::xtensor<double, 1> mu_1 {0, 0, 0};

    xt::xtensor<double, 2> Q_2 = xt::eye<double>(3);
    xt::xtensor<double, 1> mu_2 {0, 0, 0};

    std::shared_ptr<Ellipsoid3d> SF_rob = std::make_shared<Ellipsoid3d>(true, Q_1, mu_1);
    std::shared_ptr<Ellipsoid3d> SF_obs = std::make_shared<Ellipsoid3d>(true, Q_2, mu_2);
    
    xt::xtensor<double, 1> d1 {0,0,0};
    xt::xtensor<double, 1> q1 {0,0,0,1};
    xt::xtensor<double, 1> d2 {3,3,3};
    xt::xtensor<double, 1> q2 {0,0,0,1};
    
    int n_problems = 3;
    int n_threads = std::min(n_problems, (int)std::thread::hardware_concurrency());
    std::shared_ptr<Problem3dCollectionMovingObstacle> col = std::make_shared<Problem3dCollectionMovingObstacle>(n_threads);
    for (int i=0; i<n_problems; ++i){
        std::shared_ptr<EllipsoidAndEllipsoid3dPrb> prb = std::make_shared<EllipsoidAndEllipsoid3dPrb>(SF_rob, SF_obs);
        col->addProblem(prb, i, i);
    }

    xt::xtensor<double, 2> all_d_rob = xt::zeros<double>({n_problems, 3});
    xt::xtensor<double, 2> all_q_rob = xt::zeros<double>({n_problems, 4});
    xt::xtensor<double, 2> all_d_obs = xt::zeros<double>({n_problems, 3});
    xt::xtensor<double, 2> all_q_obs = xt::zeros<double>({n_problems, 4});

    for (int i=0; i<n_problems; ++i){
        xt::view(all_d_rob, i, xt::all()) = d1;
        xt::view(all_q_rob, i, xt::all()) = q1;
        xt::view(all_d_obs, i, xt::all()) = d2;
        xt::view(all_q_obs, i, xt::all()) = q2;
    }

    // int N = 1000;

    // auto start = std::chrono::high_resolution_clock::now();
    // for (int i=0; i<N; ++i){
    //     std::tuple<xt::xtensor<double, 1>, xt::xtensor<double, 2>, xt::xtensor<double, 3>> res 
    //         = col->solveGradientAndHessian(all_d_rob, all_q_rob, all_d_obs, all_q_obs);
    //     // std::cout << i << " " << std::get<0>(res) << std::endl;
    //     // std::cout << i << " " << std::get<1>(res) << std::endl;
    //     // std::cout << i << " " << std::get<2>(res) << std::endl;
    // }
    // auto end = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> diff = end - start;
    // std::cout << "solve_all took: " << diff.count() << " s\n";

    xt::xtensor<double, 1> dq_rob = xt::zeros<double>({7});
    xt::xtensor<double, 2> all_postion_rob = all_d_rob;
    xt::xtensor<double, 2> all_quat_rob = all_q_rob;
    xt::xtensor<double, 3> all_Jacobian_rob = xt::zeros<double>({n_problems, 6, 7});
    xt::xtensor<double, 2> all_dJdq_rob = xt::zeros<double>({n_problems, 6});
    xt::xtensor<double, 2> all_position_obs = all_d_obs;
    xt::xtensor<double, 2> all_quat_obs = all_q_obs;
    xt::xtensor<double, 2> all_velocity_obs = xt::zeros<double>({n_problems, 3});
    xt::xtensor<double, 2> all_omega_obs = xt::zeros<double>({n_problems, 3});
    xt::xtensor<double, 2> all_velocity_dot_obs = xt::zeros<double>({n_problems, 3});
    xt::xtensor<double, 2> all_omega_dot_obs = xt::zeros<double>({n_problems, 3});
    double alpha0 = 0;
    double gamma1 = 0;
    double gamma2 = 0;
    double compensation = 0;

    int N = 1;

    auto start = std::chrono::high_resolution_clock::now();
    for (int i=0; i<N; ++i){
        std::tuple<xt::xtensor<double, 1>, xt::xtensor<double, 1>, xt::xtensor<double, 2>, xt::xtensor<double, 1>, 
            xt::xtensor<double, 1>> res = col->getCBFConstraints(dq_rob, all_postion_rob, all_quat_rob, 
            all_Jacobian_rob, all_dJdq_rob, all_position_obs, all_quat_obs, all_velocity_obs, all_omega_obs,
            all_velocity_dot_obs, all_omega_dot_obs, alpha0, gamma1, gamma2, compensation);
        std::cout << i << " " << std::get<0>(res) << std::endl;
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << "solve_all took: " << diff.count() << " s\n";

    return 0;
}