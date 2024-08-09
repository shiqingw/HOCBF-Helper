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

void Problem3dCollection::stopAll() {
    thread_pool.stopAll();
}

std::tuple<xt::xtensor<double, 1>, xt::xtensor<double, 2>, xt::xtensor<double, 3>> Problem3dCollection::solveGradientAndHessian(
        const xt::xtensor<double, 2>& all_d, const xt::xtensor<double, 2>& all_q) {

    xt::xtensor<double, 1> all_alpha = xt::zeros<double>({n_problems});
    xt::xtensor<double, 2> all_alpha_dx = xt::zeros<double>({n_problems, 7});
    xt::xtensor<double, 3> all_alpha_dxdx = xt::zeros<double>({n_problems, 7, 7});

    for (int i = 0; i < n_problems; i++) {
        std::shared_ptr<Problem3d> problem = problems[i];
        int frame_id = frame_ids[i];
        thread_pool.enqueue([i, frame_id, problem, &all_alpha, &all_alpha_dx, &all_alpha_dxdx, &all_d, &all_q] {
            try {
                xt::xtensor<double, 1> d = xt::view(all_d, frame_id, xt::all()); // shape (3)
                xt::xtensor<double, 1> q = xt::view(all_q, frame_id, xt::all()); // shape (7)

                double alpha;
                xt::xtensor<double, 1> alpha_dx;
                xt::xtensor<double, 2> alpha_dxdx;
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

std::tuple<xt::xtensor<double, 1>, xt::xtensor<double, 2>, xt::xtensor<double, 3>, xt::xtensor<double, 1>, 
    xt::xtensor<double, 2>, xt::xtensor<double, 1>, xt::xtensor<double, 1>> Problem3dCollection::getCBFConstraints(
    const xt::xtensor<double, 1>& dq, const xt::xtensor<double, 2>& all_postion, const xt::xtensor<double, 2>& all_quat, 
    const xt::xtensor<double, 3>& all_Jacobian, const xt::xtensor<double, 2>& all_dJdq, double alpha0, double gamma1, double gamma2,
    double compensation){

    xt::xtensor<double, 1> all_h = xt::zeros<double>({n_problems});
    xt::xtensor<double, 2> all_h_dx = xt::zeros<double>({n_problems, 7});
    xt::xtensor<double, 3> all_h_dxdx = xt::zeros<double>({n_problems, 7, 7});
    xt::xtensor<double, 1> all_phi1 = xt::zeros<double>({n_problems});
    xt::xtensor<double, 2> all_actuation = xt::zeros<double>({n_problems, 7});
    xt::xtensor<double, 1> all_lb = xt::zeros<double>({n_problems});
    xt::xtensor<double, 1> all_ub = xt::zeros<double>({n_problems});

    for (int i = 0; i < n_problems; i++) {
        std::shared_ptr<Problem3d> problem = problems[i];
        int frame_id = frame_ids[i];
        thread_pool.enqueue([problem, i, frame_id, &all_h, &all_h_dx, &all_h_dxdx, &all_phi1, &all_actuation, &all_lb, &all_ub,
            &dq, &all_Jacobian, &all_postion, &all_quat, &all_dJdq, alpha0, gamma1, gamma2, compensation] {
            try {
                xt::xtensor<double, 2> J = xt::view(all_Jacobian, frame_id, xt::all(), xt::all()); // shape (6, 7)
                xt::xtensor<double, 1> d = xt::view(all_postion, frame_id, xt::all()); // shape (3, )
                xt::xtensor<double, 1> q = xt::view(all_quat, frame_id, xt::all()); // shape (4, )
                xt::xtensor<double, 1> dJdq = xt::view(all_dJdq, frame_id, xt::all()); // shape (6, )

                double alpha, h, dh, phi1, lb, ub;
                xt::xtensor<double, 1> alpha_dx, actuation, h_dx, v, dx, dquat;
                xt::xtensor<double, 2> alpha_dxdx, h_dxdx, A, Q, dA, dQ;
                std::tie(alpha, alpha_dx, alpha_dxdx) = problem->solve(d, q);
                
                v = xt::linalg::dot(J, dq); // shape (6)
                Q = getQMatrixFromQuat(q); // shape (4, 3)
                A = xt::zeros<double>({7, 6});
                auto tmp_A11_view = xt::view(A, xt::range(0, 3), xt::range(0, 3));
                xt::noalias(tmp_A11_view) = xt::eye<double>(3);

                auto tmp_A22_view = xt::view(A, xt::range(3, 7), xt::range(3, 6));
                xt::noalias(tmp_A22_view) = 0.5*Q;

                dx = xt::linalg::dot(A, v); // shape (7)
                dquat = 0.5 * xt::linalg::dot(Q, xt::view(v, xt::range(3, 6))); // shape (4)
                dQ = getdQMatrixFromdQuat(dquat); // shape (4, 3)
                dA = xt::zeros<double>({7, 6});
                auto tmp_dA22_view = xt::view(dA, xt::range(3, 7), xt::range(3, 6));
                xt::noalias(tmp_dA22_view) = 0.5*dQ;

                if (alpha != 0){
                    h = alpha - alpha0;
                    h_dx = alpha_dx;
                    h_dxdx = alpha_dxdx;

                    dh = xt::linalg::dot(h_dx, dx)(0);
                    phi1 = dh + gamma1 * h;
                    actuation = xt::linalg::dot(xt::linalg::dot(h_dx, A), J); // shape (7)
                    lb = - gamma2 * phi1 - gamma1 * dh - xt::linalg::dot(xt::linalg::dot(dx, h_dxdx), dx)(0)
                        - xt::linalg::dot(h_dx, xt::linalg::dot(dA, v))(0) - xt::linalg::dot(h_dx, xt::linalg::dot(A, dJdq))(0)
                        + compensation;
                    ub = std::numeric_limits<double>::infinity();
                } else {
                    h = std::numeric_limits<double>::infinity();
                    h_dx = xt::zeros<double>({7});
                    h_dxdx = xt::zeros<double>({7, 7});

                    phi1 = 0;
                    actuation = xt::zeros<double>({7});
                    lb = 0;
                    ub = 0;
                }

                all_h(i) = h;
                auto tmp_h_dx_view = xt::view(all_h_dx, i, xt::all());
                xt::noalias(tmp_h_dx_view) = h_dx;
                auto tmp_h_dxdx_view = xt::view(all_h_dxdx, i, xt::all(), xt::all());
                xt::noalias(tmp_h_dxdx_view) = h_dxdx;
                all_phi1(i) = phi1;
                auto tmp_actuation_view = xt::view(all_actuation, i, xt::all());
                xt::noalias(tmp_actuation_view) = actuation;
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

std::tuple<xt::xtensor<double, 1>, xt::xtensor<double, 2>, xt::xtensor<double, 3>, xt::xtensor<double, 1>, 
    xt::xtensor<double, 1>, xt::xtensor<double, 2>> Problem3dCollection::getSmoothMinCBFConstraints(
    const xt::xtensor<double, 1>& dq, const xt::xtensor<double, 2>& all_postion, const xt::xtensor<double, 2>& all_quat, 
    const xt::xtensor<double, 3>& all_Jacobian, const xt::xtensor<double, 2>& all_dJdq, double alpha0){

    xt::xtensor<double, 1> all_h = xt::zeros<double>({n_problems});
    xt::xtensor<double, 2> all_h_dx = xt::zeros<double>({n_problems, 7});
    xt::xtensor<double, 3> all_h_dxdx = xt::zeros<double>({n_problems, 7, 7});
    xt::xtensor<double, 1> all_first_order_average_scalar = xt::zeros<double>({n_problems});
    xt::xtensor<double, 1> all_second_order_average_scalar = xt::zeros<double>({n_problems});
    xt::xtensor<double, 2> all_second_order_average_vector = xt::zeros<double>({n_problems, 7});

    for (int i = 0; i < n_problems; i++) {
        std::shared_ptr<Problem3d> problem = problems[i];
        int frame_id = frame_ids[i];
        thread_pool.enqueue([problem, i, frame_id, &all_h, &all_h_dx, &all_h_dxdx, &all_first_order_average_scalar, &all_second_order_average_scalar,
            &all_second_order_average_vector, &dq, &all_Jacobian, &all_postion, &all_quat, &all_dJdq, alpha0] {
            try {
                xt::xtensor<double, 2> J = xt::view(all_Jacobian, frame_id, xt::all(), xt::all()); // shape (6, 7)
                xt::xtensor<double, 1> d = xt::view(all_postion, frame_id, xt::all()); // shape (3, )
                xt::xtensor<double, 1> q = xt::view(all_quat, frame_id, xt::all()); // shape (4, )
                xt::xtensor<double, 1> dJdq = xt::view(all_dJdq, frame_id, xt::all()); // shape (6, )

                double alpha, h, first_order_average_scalar, second_order_average_scalar;
                xt::xtensor<double, 1> alpha_dx, h_dx, v, dx, dquat, second_order_average_vector;
                xt::xtensor<double, 2> alpha_dxdx, h_dxdx, A, Q, dA, dQ;
                std::tie(alpha, alpha_dx, alpha_dxdx) = problem->solve(d, q);
                
                v = xt::linalg::dot(J, dq); // shape (6)
                Q = getQMatrixFromQuat(q); // shape (4, 3)
                A = xt::zeros<double>({7, 6});
                auto tmp_A11_view = xt::view(A, xt::range(0, 3), xt::range(0, 3));
                xt::noalias(tmp_A11_view) = xt::eye<double>(3);

                auto tmp_A22_view = xt::view(A, xt::range(3, 7), xt::range(3, 6));
                xt::noalias(tmp_A22_view) = 0.5*Q;

                dx = xt::linalg::dot(A, v); // shape (7)
                dquat = 0.5 * xt::linalg::dot(Q, xt::view(v, xt::range(3, 6))); // shape (4)
                dQ = getdQMatrixFromdQuat(dquat); // shape (4, 3)
                dA = xt::zeros<double>({7, 6});
                auto tmp_dA22_view = xt::view(dA, xt::range(3, 7), xt::range(3, 6));
                xt::noalias(tmp_dA22_view) = 0.5*dQ;

                if (alpha != 0){
                    h = alpha - alpha0;
                    h_dx = alpha_dx;
                    h_dxdx = alpha_dxdx;

                    first_order_average_scalar = xt::linalg::dot(h_dx, dx)(0);

                    second_order_average_scalar = xt::linalg::dot(xt::linalg::dot(dx, h_dxdx), dx)(0)
                        + xt::linalg::dot(h_dx, xt::linalg::dot(dA, v))(0) + xt::linalg::dot(h_dx, xt::linalg::dot(A, dJdq))(0);

                    second_order_average_vector = xt::linalg::dot(xt::linalg::dot(h_dx, A), J); // shape (7)
                } else {
                    h = std::numeric_limits<double>::infinity();
                    h_dx = xt::zeros<double>({7});
                    h_dxdx = xt::zeros<double>({7, 7});
                    
                    first_order_average_scalar = 0;
                    second_order_average_scalar = 0;
                    second_order_average_vector = xt::zeros<double>({7});
                }

                all_h(i) = h;
                auto tmp_h_dx_view = xt::view(all_h_dx, i, xt::all());
                xt::noalias(tmp_h_dx_view) = h_dx;
                auto tmp_h_dxdx_view = xt::view(all_h_dxdx, i, xt::all(), xt::all());
                xt::noalias(tmp_h_dxdx_view) = h_dxdx;
                all_first_order_average_scalar(i) = first_order_average_scalar;
                all_second_order_average_scalar(i) = second_order_average_scalar;
                auto tmp_second_order_average_vector_view = xt::view(all_second_order_average_vector, i, xt::all());
                xt::noalias(tmp_second_order_average_vector_view) = second_order_average_vector;

            } catch (const std::exception& e) {
                std::cerr << "Exception in thread " << i << ": " << e.what() << std::endl;
            }
        });
    }

    thread_pool.wait();
    
    return std::make_tuple(all_h, all_h_dx, all_h_dxdx, all_first_order_average_scalar, all_second_order_average_scalar, all_second_order_average_vector);
}