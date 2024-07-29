#include <iostream>
#include <memory>
#include <xtensor/xtensor.hpp>
#include <xtensor/xio.hpp>
#include <chrono>
#include <csignal>

#include "ellipsoidAndLogSumExp2dPrb.hpp"
#include "ellipsoidAndHyperplane2dPrb.hpp"
#include "problem2dCollection.hpp"
#include "scalingFunctionsHelper/ellipsoid2d.hpp"
#include "scalingFunctionsHelper/logSumExp2d.hpp"
#include "scalingFunctionsHelper/hyperplane2d.hpp"

std::shared_ptr<Problem2dCollection> col;

void (*oldHandler)(int) = nullptr;

void signalHandler(int signum) {
    std::cout << "\nInterrupt signal (" << signum << ") received. Stopping all problems." << std::endl;
    
    // Call your cleanup code here
    if (col) {
        col->stopAll();
    }

    // If there was an old signal handler, call it
    if (oldHandler) {
        oldHandler(signum);
    }

    // Exit or do other final cleanup
    std::exit(signum);
}

int main() {
    oldHandler = std::signal(SIGINT, signalHandler);

    xt::xtensor<double, 2> Q_d = xt::eye<double>(2);
    xt::xtensor<double, 1> mu_d {4, 1};

    xt::xtensor<double, 2> A_d {{1,0},
                                {-1,0},
                                {0,1},
                                {0,-1}};
    xt::xtensor<double, 1> b_d = -xt::ones<double>({4});
    double kappa_d = 20.0;
    xt::xtensor<double, 2> vertices{{1,1},
                                    {1,-1},
                                    {-1,1},
                                    {-1,-1}};

    xt::xtensor<double, 1> H_a {1,0};
    double H_b = -1.0;

    std::shared_ptr<Ellipsoid2d> SF_rob(new Ellipsoid2d(true, Q_d, xt::zeros<double>({2})));
    std::shared_ptr<LogSumExp2d> SF_obs(new LogSumExp2d(false, A_d, b_d, kappa_d));
    std::shared_ptr<Hyperplane2d> SF_obs_hyper(new Hyperplane2d(false, H_a, H_b));
    
    int n_repeat = 10;
    int n_problems = 2*n_repeat;
    int n_threads = 9;
    col = std::make_shared<Problem2dCollection>(n_threads);
    for (int i=0; i<n_repeat; ++i){
        std::shared_ptr<EllipsoidAndLogSumExp2dPrb> prb = std::make_shared<EllipsoidAndLogSumExp2dPrb>(SF_rob, SF_obs, vertices);
        col->addProblem(prb, i);
    }

    for (int i=0; i<n_repeat; ++i){
        std::shared_ptr<EllipsoidAndHyperplane2dPrb> prb = std::make_shared<EllipsoidAndHyperplane2dPrb>(SF_rob, SF_obs_hyper);
        col->addProblem(prb, i + n_repeat);
    }

    xt::xtensor<double, 2> all_d = xt::zeros<double>({n_problems, 2});
    for (int i=0; i<n_problems; ++i){
        all_d(i, 0) = mu_d(0);
        all_d(i, 1) = mu_d(1);
    }
    xt::xtensor<double, 1> all_theta = xt::zeros<double>({n_problems});

    // int N = std::stoi(argv[1]);
    int N = 10000;

    auto start = std::chrono::high_resolution_clock::now();
    for (int i=0; i<N; ++i){
        std::cout << i << " " << col->thread_pool.stop << std::endl;
        std::tuple<xt::xtensor<double, 1>, xt::xtensor<double, 2>, xt::xtensor<double, 3>> res = col->solveGradientAndHessian(all_d, all_theta);
        // std::cout << i << " " << std::get<0>(res) << std::endl;
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << "solve_all took: " << diff.count() << " s\n";

    return 0;
}