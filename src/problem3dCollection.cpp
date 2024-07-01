#include "problem3dCollection.hpp"

Problem3dCollection::Problem3dCollection(size_t n_threads_) : thread_pool(n_threads_) {
    n_threads = n_threads_;
}

Problem3dCollection::~Problem3dCollection() {
    thread_pool.wait();
}

void Problem3dCollection::add_problem(std::shared_ptr<Problem3d> problem) {
    problems.push_back(problem);
    n_problems++;
}

std::tuple<xt::xarray<double>, xt::xarray<double>, xt::xarray<double>> Problem3dCollection::solve_all(
    const xt::xarray<double>& all_d, const xt::xarray<double>& all_q) {

    xt::xarray<double> all_alpha = xt::zeros<double>({n_problems});
    xt::xarray<double> all_alpha_dx = xt::zeros<double>({n_problems, 7});
    xt::xarray<double> all_alpha_dxdx = xt::zeros<double>({n_problems, 7, 7});

    for (int i = 0; i < n_problems; i++) {
        thread_pool.enqueue([this, i, &all_alpha, &all_alpha_dx, &all_alpha_dxdx, all_d, all_q] {
            try {
                auto problem = problems[i];
                double alpha;
                xt::xarray<double> alpha_dx, alpha_dxdx;
                xt::xarray<double> d = xt::view(all_d, i, xt::all());
                xt::xarray<double> q = xt::view(all_q, i, xt::all());
                std::tie(alpha, alpha_dx, alpha_dxdx) = problem->solve(d, q);
                all_alpha[i] = alpha;
                xt::view(all_alpha_dx, i, xt::all()) = alpha_dx;
                xt::view(all_alpha_dxdx, i, xt::all(), xt::all()) = alpha_dxdx;
            } catch (const std::exception& e) {
                std::cerr << "Exception in thread " << i << ": " << e.what() << std::endl;
            }
        });
    }

    thread_pool.wait();

    return std::make_tuple(all_alpha, all_alpha_dx, all_alpha_dxdx);
}

void Problem3dCollection::wait_all() {
    thread_pool.wait();
}

