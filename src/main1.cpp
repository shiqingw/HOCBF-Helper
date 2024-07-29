#include <iostream>
#include <memory>
#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <chrono>

#include "ellipsoidAndLogSumExp3dPrb.hpp"
#include "problem3dCollection.hpp"

// int main(int argc, char* argv[]) {
int main() {
    xt::xarray<double> Q_d = xt::eye<double>(3);
    xt::xarray<double> mu_d {4, 1, 1};
    xt::xarray<double> A_d {{1,0,0},
              {-1,0,0},
              {0,1,0},
              {0,-1,0},
              {0,0,1},
              {0,0,-1}};
    xt::xarray<double> b_d = -xt::ones<double>({6});
    double kappa_d = 10.0;
    xt::xarray<double> vertices{{1,1,1},
                                {1,1,-1},
                                {1,-1,1},
                                {1,-1,-1},
                                {-1,1,1},
                                {-1,1,-1},
                                {-1,-1,1},
                                {-1,-1,-1}};

    std::shared_ptr<Ellipsoid3d> SF_rob(new Ellipsoid3d(true, Q_d, xt::zeros<double>({3})));
    std::shared_ptr<LogSumExp3d> SF_obs(new LogSumExp3d(false, A_d, b_d, kappa_d));


    xt::xarray<double> q {0,0,0,1};
    
    int n_problems = 9;
    int n_threads = 18;
    std::shared_ptr<Problem3dCollection> col = std::make_shared<Problem3dCollection>(n_threads);
    for (int i=0; i<n_problems; ++i){
        // std::shared_ptr<Ellipsoid3d> SF_rob(new Ellipsoid3d(true, Q_d, xt::zeros<double>({3})));
        // std::shared_ptr<LogSumExp3d> SF_obs(new LogSumExp3d(false, A_d, b_d, kappa_d));
        std::shared_ptr<ElliposoidAndLogSumExp3dPrb> prb = std::make_shared<ElliposoidAndLogSumExp3dPrb>(SF_rob, SF_obs, vertices);
        col->addProblem(prb, i);
    }

    xt::xarray<double> all_d = xt::zeros<double>({n_problems, 3});
    xt::xarray<double> all_q = xt::zeros<double>({n_problems, 4});
    for (int i=0; i<n_problems; ++i){
        xt::view(all_d, i, xt::all()) = mu_d;
        all_q(i, 3) = 1;
    }

    xt::xarray<double> dq = xt::zeros<double>({7});
    xt::xarray<double> all_Jacobian = xt::zeros<double>({n_problems, 6, 7});
    xt::xarray<double> all_dJdq = xt::zeros<double>({n_problems, 6});
    double alpha0 = 0.1;
    double gamma1 = 0.1;
    double gamma2 = 0.1;
    double compensation = 0.1;

    // int N = std::stoi(argv[1]);
    int N = 1000;

    auto start = std::chrono::high_resolution_clock::now();
    for (int i=0; i<N; ++i){
        // std::tuple<xt::xarray<double>, xt::xarray<double>, xt::xarray<double>> res = col->solveGradientAndHessian(all_d, all_q);
        std::tuple<xt::xarray<double>, xt::xarray<double>, xt::xarray<double>, xt::xarray<double>, 
            xt::xarray<double>, xt::xarray<double>, xt::xarray<double>> res =col->getCBFConstraints(dq, 
            all_Jacobian, all_d, all_q, all_dJdq, alpha0, gamma1, gamma2, compensation);
        // std::cout << i << " " << std::get<0>(res) << std::endl;
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << "solve_all took: " << diff.count() << " s\n";

    return 0;
}