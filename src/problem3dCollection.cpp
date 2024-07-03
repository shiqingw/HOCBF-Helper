#include "problem3dCollection.hpp"

Problem3dCollection::Problem3dCollection(size_t n_threads_) : thread_pool(n_threads_) {
    n_threads = n_threads_;
}

Problem3dCollection::~Problem3dCollection() {
    thread_pool.wait();
}

void Problem3dCollection::addProblem(std::shared_ptr<Problem3d> problem, int frame_id) {
    problems.push_back(problem);
    frame_ids.push_back(frame_id);
    n_problems++;
}

void Problem3dCollection::waitAll() {
    thread_pool.wait();
}

std::tuple<xt::xarray<double>, xt::xarray<double>, xt::xarray<double>> Problem3dCollection::solveGradientAndHessian(
    const xt::xarray<double>& all_d, const xt::xarray<double>& all_q) {

    xt::xarray<double> all_alpha = xt::zeros<double>({n_problems});
    xt::xarray<double> all_alpha_dx = xt::zeros<double>({n_problems, 7});
    xt::xarray<double> all_alpha_dxdx = xt::zeros<double>({n_problems, 7, 7});

    for (int i = 0; i < n_problems; i++) {
        std::shared_ptr<Problem3d> problem = problems[i];
        thread_pool.enqueue([i, problem, &all_alpha, &all_alpha_dx, &all_alpha_dxdx, &all_d, &all_q] {
            try {
                xt::xarray<double> d = xt::view(all_d, i, xt::all()); // shape (3)
                xt::xarray<double> q = xt::view(all_q, i, xt::all()); // shape (7)

                double alpha;
                xt::xarray<double> alpha_dx, alpha_dxdx;
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

std::tuple<xt::xarray<double>, xt::xarray<double>, xt::xarray<double>, xt::xarray<double>, 
    xt::xarray<double>, xt::xarray<double>, xt::xarray<double>> Problem3dCollection::getCBFConstraints(
    const xt::xarray<double>& dq, const xt::xarray<double>& all_postion, const xt::xarray<double>& all_quat, 
    const xt::xarray<double>& all_Jacobian, const xt::xarray<double>& all_dJdq, double alpha0, double gamma1, double gamma2,
    double compensation){

    xt::xarray<double> all_h = xt::zeros<double>({n_problems});
    xt::xarray<double> all_h_dx = xt::zeros<double>({n_problems, 7});
    xt::xarray<double> all_h_dxdx = xt::zeros<double>({n_problems, 7, 7});
    xt::xarray<double> all_phi1 = xt::zeros<double>({n_problems});
    xt::xarray<double> all_actuation = xt::zeros<double>({n_problems, 7});
    xt::xarray<double> all_lb = xt::zeros<double>({n_problems});
    xt::xarray<double> all_ub = xt::zeros<double>({n_problems});

    for (int i = 0; i < n_problems; i++) {
        std::shared_ptr<Problem3d> problem = problems[i];
        int frame_id = frame_ids[i];
        thread_pool.enqueue([problem, i, frame_id, &all_h, &all_h_dx, &all_h_dxdx, &all_phi1, &all_actuation, &all_lb, &all_ub,
            &dq, &all_Jacobian, &all_postion, &all_quat, &all_dJdq, alpha0, gamma1, gamma2, compensation] {
            try {
                xt::xarray<double> J = xt::view(all_Jacobian, frame_id, xt::all(), xt::all()); // shape (6, 7)
                xt::xarray<double> d = xt::view(all_postion, frame_id, xt::all()); // shape (3)
                xt::xarray<double> q = xt::view(all_quat, frame_id, xt::all()); // shape (4)
                xt::xarray<double> dJdq = xt::view(all_dJdq, frame_id, xt::all()); // shape (7)

                double alpha, h, dh, phi1, lb, ub;
                xt::xarray<double> alpha_dx, alpha_dxdx, actuation, h_dx, h_dxdx, v, A, Q, dx, dquat, dA, dQ;
                std::tie(alpha, alpha_dx, alpha_dxdx) = problem->solve(d, q);

                h = alpha - alpha0;
                h_dx = alpha_dx;
                h_dxdx = alpha_dxdx;
                
                v = xt::linalg::dot(J, dq); // shape (6)
                Q = getQMatrixFromQuat(q); // shape (4, 3)
                A = xt::zeros<double>({7, 6});
                xt::view(A, xt::range(0, 3), xt::range(0, 3)) = xt::eye<double>(3);
                xt::view(A, xt::range(3, 7), xt::range(3, 6)) = 0.5*Q;
                dx = xt::linalg::dot(A, v); // shape (7)
                dquat = 0.5 * xt::linalg::dot(Q, xt::view(v, xt::range(3, 6))); // shape (4)
                dQ = getdQMatrixFromdQuat(dquat); // shape (4, 3)
                dA = xt::zeros<double>({7, 6});
                xt::view(dA, xt::range(3, 7), xt::range(3, 6)) = 0.5*dQ;

                if (alpha != 0){
                    dh = xt::linalg::dot(h_dx, dx)(0);
                    phi1 = dh + gamma1 * h;
                    actuation = xt::linalg::dot(xt::linalg::dot(h_dx, A), J); // shape (7)
                    lb = - gamma2 * phi1 - gamma1 * dh - xt::linalg::dot(xt::linalg::dot(dx, h_dxdx), dx)(0)
                        - xt::linalg::dot(h_dx, xt::linalg::dot(dA, v))(0) - xt::linalg::dot(h_dx, xt::linalg::dot(A, dJdq))(0)
                        + compensation;
                    ub = std::numeric_limits<double>::infinity();
                } else {
                    phi1 = 0;
                    actuation = xt::zeros<double>({7});
                    lb = 0;
                    ub = 0;
                }

                all_h(i) = h;
                xt::view(all_h_dx, i, xt::all()) = h_dx;
                xt::view(all_h_dxdx, i, xt::all(), xt::all()) = h_dxdx;
                all_phi1(i) = phi1;
                xt::view(all_actuation, i, xt::all()) = actuation;
                all_lb(i) = lb;
                all_ub(i) = ub;

            } catch (const std::exception& e) {
                std::cerr << "Exception in thread " << i << ": " << e.what() << std::endl;
            }
        });
    }

    thread_pool.wait();
    
    return std::make_tuple(all_h, all_h_dx, all_h_dxdx, all_phi1, all_actuation, all_lb, all_ub);
}
