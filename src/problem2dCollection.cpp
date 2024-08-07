#include "problem2dCollection.hpp"

Problem2dCollection::Problem2dCollection(size_t n_threads_) : thread_pool(n_threads_) {
    n_threads = n_threads_;
}

Problem2dCollection::~Problem2dCollection() {
    thread_pool.wait();
}

void Problem2dCollection::addProblem(std::shared_ptr<Problem2d> problem, int frame_id) {
    problems.push_back(problem);
    frame_ids.push_back(frame_id);
    n_problems++;
}

void Problem2dCollection::waitAll() {
    thread_pool.wait();
}

void Problem2dCollection::stopAll() {
    thread_pool.stopAll();
}

std::tuple<xt::xtensor<double, 1>, xt::xtensor<double, 2>, xt::xtensor<double, 3>> Problem2dCollection::solveGradientAndHessian(
        const xt::xtensor<double, 2>& all_d, const xt::xtensor<double, 1>& all_theta) {

    xt::xtensor<double, 1> all_alpha = xt::zeros<double>({n_problems});
    xt::xtensor<double, 2> all_alpha_dx = xt::zeros<double>({n_problems, 3});
    xt::xtensor<double, 3> all_alpha_dxdx = xt::zeros<double>({n_problems, 3, 3});

    for (int i = 0; i < n_problems; i++) {
        std::shared_ptr<Problem2d> problem = problems[i];
        thread_pool.enqueue([i, problem, &all_alpha, &all_alpha_dx, &all_alpha_dxdx, &all_d, &all_theta] {
            try {
                xt::xtensor<double, 1> d = xt::view(all_d, i, xt::all()); // shape (3)
                double theta = all_theta(i); // scalar

                double alpha;
                xt::xtensor<double, 1> alpha_dx;
                xt::xtensor<double, 2> alpha_dxdx;
                std::tie(alpha, alpha_dx, alpha_dxdx) = problem->solve(d, theta);
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
    xt::xtensor<double, 2>, xt::xtensor<double, 1>, xt::xtensor<double, 1>> Problem2dCollection::getCBFConstraints(
    const xt::xtensor<double, 2>& all_postion, const xt::xtensor<double, 1>& all_theta, const xt::xtensor<double, 2>& all_dx,
    double alpha0, double gamma1, double gamma2, double compensation){

    xt::xtensor<double, 1> all_h = xt::zeros<double>({n_problems});
    xt::xtensor<double, 2> all_h_dx = xt::zeros<double>({n_problems, 3});
    xt::xtensor<double, 3> all_h_dxdx = xt::zeros<double>({n_problems, 3, 3});
    xt::xtensor<double, 1> all_phi1 = xt::zeros<double>({n_problems});
    xt::xtensor<double, 2> all_actuation = xt::zeros<double>({n_problems, 3});
    xt::xtensor<double, 1> all_lb = xt::zeros<double>({n_problems});
    xt::xtensor<double, 1> all_ub = xt::zeros<double>({n_problems});

    for (int i = 0; i < n_problems; i++) {
        std::shared_ptr<Problem2d> problem = problems[i];
        int frame_id = frame_ids[i];
        thread_pool.enqueue([problem, i, frame_id, &all_h, &all_h_dx, &all_h_dxdx, &all_phi1, &all_actuation, &all_lb, &all_ub,
            &all_postion, &all_theta, &all_dx, alpha0, gamma1, gamma2, compensation] {
            try {
                xt::xtensor<double, 1> d = xt::view(all_postion, frame_id, xt::keep(0,1)); // shape (2, )
                double theta = all_theta(frame_id); // scalar

                double alpha, h, dh, phi1, lb, ub;
                xt::xtensor<double, 1> alpha_dx, actuation, h_dx, v, dx;
                xt::xtensor<double, 2> alpha_dxdx, h_dxdx;
                std::tie(alpha, alpha_dx, alpha_dxdx) = problem->solve(d, theta);
                
                dx = xt::view(all_dx, frame_id, xt::all()); // shape (3)
                
                if (alpha != 0){
                    h = alpha - alpha0;
                    h_dx = alpha_dx;
                    h_dxdx = alpha_dxdx;

                    dh = xt::linalg::dot(h_dx, dx)(0);
                    phi1 = dh + gamma1 * h;
                    actuation = h_dx; // shape (3)
                    lb = - gamma2 * phi1 - gamma1 * dh - xt::linalg::dot(xt::linalg::dot(dx, h_dxdx), dx)(0)
                        + compensation;
                    ub = std::numeric_limits<double>::infinity();
                    
                } else {
                    h = std::numeric_limits<double>::infinity();
                    h_dx = xt::zeros<double>({3});
                    h_dxdx = xt::zeros<double>({3, 3});

                    phi1 = 0;
                    actuation = xt::zeros<double>({3});
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
    xt::xtensor<double, 2>, xt::xtensor<double, 1>, xt::xtensor<double, 1>> Problem2dCollection::getCBFConstraintsFixedOrientation(
    const xt::xtensor<double, 2>& all_postion, const xt::xtensor<double, 1>& all_theta, const xt::xtensor<double, 2>& all_dx,
    double alpha0, double gamma1, double gamma2, double compensation){

    xt::xtensor<double, 1> all_h = xt::zeros<double>({n_problems});
    xt::xtensor<double, 2> all_h_dx = xt::zeros<double>({n_problems, 3});
    xt::xtensor<double, 3> all_h_dxdx = xt::zeros<double>({n_problems, 3, 3});
    xt::xtensor<double, 1> all_phi1 = xt::zeros<double>({n_problems});
    xt::xtensor<double, 2> all_actuation = xt::zeros<double>({n_problems, 3});
    xt::xtensor<double, 1> all_lb = xt::zeros<double>({n_problems});
    xt::xtensor<double, 1> all_ub = xt::zeros<double>({n_problems});

    for (int i = 0; i < n_problems; i++) {
        std::shared_ptr<Problem2d> problem = problems[i];
        int frame_id = frame_ids[i];
        thread_pool.enqueue([problem, i, frame_id, &all_h, &all_h_dx, &all_h_dxdx, &all_phi1, &all_actuation, &all_lb, &all_ub,
            &all_postion, &all_theta, &all_dx, alpha0, gamma1, gamma2, compensation] {
            try {
                xt::xtensor<double, 1> d = xt::view(all_postion, frame_id, xt::keep(0,1)); // shape (2, )
                double theta = all_theta(frame_id); // scalar

                double alpha, h, dh, phi1, lb, ub;
                xt::xtensor<double, 1> alpha_dx, actuation, h_dx, v, dx;
                xt::xtensor<double, 2> alpha_dxdx, h_dxdx;
                std::tie(alpha, alpha_dx, alpha_dxdx) = problem->solve(d, theta);

                // Set gradients and hessians to zero for the orientation part
                alpha_dx(2) = 0;
                alpha_dxdx(0, 2) = 0;
                alpha_dxdx(1, 2) = 0;
                alpha_dxdx(2, 0) = 0;
                alpha_dxdx(2, 1) = 0;
                alpha_dxdx(2, 2) = 0;
                
                dx = xt::view(all_dx, frame_id, xt::all()); // shape (3)
                
                if (alpha != 0){
                    h = alpha - alpha0;
                    h_dx = alpha_dx;
                    h_dxdx = alpha_dxdx;

                    dh = xt::linalg::dot(h_dx, dx)(0);
                    phi1 = dh + gamma1 * h;
                    actuation = h_dx; // shape (3)
                    lb = - gamma2 * phi1 - gamma1 * dh - xt::linalg::dot(xt::linalg::dot(dx, h_dxdx), dx)(0)
                        + compensation;
                    ub = std::numeric_limits<double>::infinity();
                    
                } else {
                    h = std::numeric_limits<double>::infinity();
                    h_dx = xt::zeros<double>({3});
                    h_dxdx = xt::zeros<double>({3, 3});

                    phi1 = 0;
                    actuation = xt::zeros<double>({3});
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
    xt::xtensor<double, 1>, xt::xtensor<double, 2>> Problem2dCollection::getSmoothMinCBFConstraints(
    const xt::xtensor<double, 2>& all_postion, const xt::xtensor<double, 1>& all_theta, const xt::xtensor<double, 2>& all_dx,
    double alpha0){

    xt::xtensor<double, 1> all_h = xt::zeros<double>({n_problems});
    xt::xtensor<double, 2> all_h_dx = xt::zeros<double>({n_problems, 3});
    xt::xtensor<double, 3> all_h_dxdx = xt::zeros<double>({n_problems, 3, 3});
    xt::xtensor<double, 1> all_first_order_average_scalar = xt::zeros<double>({n_problems});
    xt::xtensor<double, 1> all_second_order_average_scalar = xt::zeros<double>({n_problems});
    xt::xtensor<double, 2> all_second_order_average_vector = xt::zeros<double>({n_problems, 3});

    for (int i = 0; i < n_problems; i++) {
        std::shared_ptr<Problem2d> problem = problems[i];
        int frame_id = frame_ids[i];
        thread_pool.enqueue([problem, i, frame_id, &all_h, &all_h_dx, &all_h_dxdx, &all_first_order_average_scalar, &all_second_order_average_scalar,
            &all_second_order_average_vector, &all_postion, &all_theta, &all_dx, alpha0] {
            try {
                xt::xtensor<double, 1> d = xt::view(all_postion, frame_id, xt::keep(0,1)); // shape (2, )
                double theta = all_theta(frame_id); // scalar

                double alpha, h, first_order_average_scalar, second_order_average_scalar;
                xt::xtensor<double, 1> alpha_dx, h_dx, dx, second_order_average_vector;
                xt::xtensor<double, 2> alpha_dxdx, h_dxdx;
                std::tie(alpha, alpha_dx, alpha_dxdx) = problem->solve(d, theta);
            
                dx = xt::view(all_dx, frame_id, xt::all()); // shape (3)
                
                if (alpha != 0){
                    h = alpha - alpha0;
                    h_dx = alpha_dx;
                    h_dxdx = alpha_dxdx;

                    first_order_average_scalar = xt::linalg::dot(h_dx, dx)(0);
                    second_order_average_scalar = xt::linalg::dot(xt::linalg::dot(dx, h_dxdx), dx)(0);
                    second_order_average_vector = h_dx; // shape (3)
                } else {
                    h = std::numeric_limits<double>::infinity();
                    h_dx = xt::zeros<double>({3});
                    h_dxdx = xt::zeros<double>({3, 3});
                    
                    first_order_average_scalar = 0;
                    second_order_average_scalar = 0;
                    second_order_average_vector = xt::zeros<double>({3});
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

std::tuple<xt::xtensor<double, 1>, xt::xtensor<double, 2>, xt::xtensor<double, 3>, xt::xtensor<double, 1>, 
    xt::xtensor<double, 1>, xt::xtensor<double, 2>> Problem2dCollection::getSmoothMinCBFConstraintsFixedOrientation(
    const xt::xtensor<double, 2>& all_postion, const xt::xtensor<double, 1>& all_theta, const xt::xtensor<double, 2>& all_dx,
    double alpha0){

    xt::xtensor<double, 1> all_h = xt::zeros<double>({n_problems});
    xt::xtensor<double, 2> all_h_dx = xt::zeros<double>({n_problems, 3});
    xt::xtensor<double, 3> all_h_dxdx = xt::zeros<double>({n_problems, 3, 3});
    xt::xtensor<double, 1> all_first_order_average_scalar = xt::zeros<double>({n_problems});
    xt::xtensor<double, 1> all_second_order_average_scalar = xt::zeros<double>({n_problems});
    xt::xtensor<double, 2> all_second_order_average_vector = xt::zeros<double>({n_problems, 3});

    for (int i = 0; i < n_problems; i++) {
        std::shared_ptr<Problem2d> problem = problems[i];
        int frame_id = frame_ids[i];
        thread_pool.enqueue([problem, i, frame_id, &all_h, &all_h_dx, &all_h_dxdx, &all_first_order_average_scalar, &all_second_order_average_scalar,
            &all_second_order_average_vector, &all_postion, &all_theta, &all_dx, alpha0] {
            try {
                xt::xtensor<double, 1> d = xt::view(all_postion, frame_id, xt::keep(0,1)); // shape (2, )
                double theta = all_theta(frame_id); // scalar

                double alpha, h, first_order_average_scalar, second_order_average_scalar;
                xt::xtensor<double, 1> alpha_dx, h_dx, dx, second_order_average_vector;
                xt::xtensor<double, 2> alpha_dxdx, h_dxdx;
                std::tie(alpha, alpha_dx, alpha_dxdx) = problem->solve(d, theta);

                // Set gradients and hessians to zero for the orientation part
                alpha_dx(2) = 0;
                alpha_dxdx(0, 2) = 0;
                alpha_dxdx(1, 2) = 0;
                alpha_dxdx(2, 0) = 0;
                alpha_dxdx(2, 1) = 0;
                alpha_dxdx(2, 2) = 0;
            
                dx = xt::view(all_dx, frame_id, xt::all()); // shape (3)
                
                if (alpha != 0){
                    h = alpha - alpha0;
                    h_dx = alpha_dx;
                    h_dxdx = alpha_dxdx;

                    first_order_average_scalar = xt::linalg::dot(h_dx, dx)(0);
                    second_order_average_scalar = xt::linalg::dot(xt::linalg::dot(dx, h_dxdx), dx)(0);
                    second_order_average_vector = h_dx; // shape (3)
                } else {
                    h = std::numeric_limits<double>::infinity();
                    h_dx = xt::zeros<double>({3});
                    h_dxdx = xt::zeros<double>({3, 3});
                    
                    first_order_average_scalar = 0;
                    second_order_average_scalar = 0;
                    second_order_average_vector = xt::zeros<double>({3});
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


// std::tuple<xt::xtensor<double, 1>, xt::xtensor<double, 2>, xt::xtensor<double, 3>, xt::xtensor<double, 1>, 
//     xt::xtensor<double, 2>, xt::xtensor<double, 1>, xt::xtensor<double, 1>> Problem2dCollection::getCBFConstraintsOld(
//     const xt::xtensor<double, 1>& dq, const xt::xtensor<double, 2>& all_postion, const xt::xtensor<double, 1>& all_theta, 
//     const xt::xtensor<double, 3>& all_Jacobian, const xt::xtensor<double, 2>& all_dJdq, double alpha0, double gamma1, double gamma2,
//     double compensation){

//     xt::xtensor<double, 1> all_h = xt::zeros<double>({n_problems});
//     xt::xtensor<double, 2> all_h_dx = xt::zeros<double>({n_problems, 3});
//     xt::xtensor<double, 3> all_h_dxdx = xt::zeros<double>({n_problems, 3, 3});
//     xt::xtensor<double, 1> all_phi1 = xt::zeros<double>({n_problems});
//     xt::xtensor<double, 2> all_actuation = xt::zeros<double>({n_problems, 7});
//     xt::xtensor<double, 1> all_lb = xt::zeros<double>({n_problems});
//     xt::xtensor<double, 1> all_ub = xt::zeros<double>({n_problems});

//     for (int i = 0; i < n_problems; i++) {
//         std::shared_ptr<Problem2d> problem = problems[i];
//         int frame_id = frame_ids[i];
//         thread_pool.enqueue([problem, i, frame_id, &all_h, &all_h_dx, &all_h_dxdx, &all_phi1, &all_actuation, &all_lb, &all_ub,
//             &dq, &all_Jacobian, &all_postion, &all_theta, &all_dJdq, alpha0, gamma1, gamma2, compensation] {
//             try {
//                 xt::xtensor<double, 2> J = xt::view(all_Jacobian, frame_id, xt::keep(0,1,5), xt::all()); // shape (3, 7)
//                 xt::xtensor<double, 1> d = xt::view(all_postion, frame_id, xt::keep(0,1)); // shape (2, )
//                 double theta = all_theta(frame_id); // scalar
//                 xt::xtensor<double, 1> dJdq = xt::view(all_dJdq, frame_id, xt::keep(0,1,5)); // shape (3, )

//                 double alpha, h, dh, phi1, lb, ub;
//                 xt::xtensor<double, 1> alpha_dx, actuation, h_dx, v, dx;
//                 xt::xtensor<double, 2> alpha_dxdx, h_dxdx;
//                 std::tie(alpha, alpha_dx, alpha_dxdx) = problem->solve(d, theta);
                
//                 dx = xt::linalg::dot(J, dq); // shape (3)
                
//                 if (alpha != 0){
//                     h = alpha - alpha0;
//                     h_dx = alpha_dx;
//                     h_dxdx = alpha_dxdx;

//                     dh = xt::linalg::dot(h_dx, dx)(0);
//                     phi1 = dh + gamma1 * h;
//                     actuation = xt::linalg::dot(h_dx, J); // shape (7)
//                     lb = - gamma2 * phi1 - gamma1 * dh - xt::linalg::dot(xt::linalg::dot(dx, h_dxdx), dx)(0)
//                         - xt::linalg::dot(h_dx, dJdq)(0) + compensation;
//                     ub = std::numeric_limits<double>::infinity();
                    
//                 } else {
//                     h = std::numeric_limits<double>::infinity();
//                     h_dx = xt::zeros<double>({3});
//                     h_dxdx = xt::zeros<double>({3, 3});

//                     phi1 = 0;
//                     actuation = xt::zeros<double>({7});
//                     lb = 0;
//                     ub = 0;
//                 }

//                 all_h(i) = h;
//                 auto tmp_h_dx_view = xt::view(all_h_dx, i, xt::all());
//                 xt::noalias(tmp_h_dx_view) = h_dx;
//                 auto tmp_h_dxdx_view = xt::view(all_h_dxdx, i, xt::all(), xt::all());
//                 xt::noalias(tmp_h_dxdx_view) = h_dxdx;
//                 all_phi1(i) = phi1;
//                 auto tmp_actuation_view = xt::view(all_actuation, i, xt::all());
//                 xt::noalias(tmp_actuation_view) = actuation;
//                 all_lb(i) = lb;
//                 all_ub(i) = ub;

//             } catch (const std::exception& e) {
//                 std::cerr << "Exception in thread " << i << ": " << e.what() << std::endl;
//             }
//         });
//     }

//     thread_pool.wait();
    
//     return std::make_tuple(all_h, all_h_dx, all_h_dxdx, all_phi1, all_actuation, all_lb, all_ub);
// }

// std::tuple<xt::xtensor<double, 1>, xt::xtensor<double, 2>, xt::xtensor<double, 3>, xt::xtensor<double, 1>, 
//     xt::xtensor<double, 2>, xt::xtensor<double, 1>, xt::xtensor<double, 1>> Problem2dCollection::getCBFConstraintsFixedOrientationOld(
//     const xt::xtensor<double, 1>& dq, const xt::xtensor<double, 2>& all_postion, const xt::xtensor<double, 1>& all_theta, 
//     const xt::xtensor<double, 3>& all_Jacobian, const xt::xtensor<double, 2>& all_dJdq, double alpha0, double gamma1, double gamma2,
//     double compensation){

//     xt::xtensor<double, 1> all_h = xt::zeros<double>({n_problems});
//     xt::xtensor<double, 2> all_h_dx = xt::zeros<double>({n_problems, 3});
//     xt::xtensor<double, 3> all_h_dxdx = xt::zeros<double>({n_problems, 3, 3});
//     xt::xtensor<double, 1> all_phi1 = xt::zeros<double>({n_problems});
//     xt::xtensor<double, 2> all_actuation = xt::zeros<double>({n_problems, 7});
//     xt::xtensor<double, 1> all_lb = xt::zeros<double>({n_problems});
//     xt::xtensor<double, 1> all_ub = xt::zeros<double>({n_problems});

//     for (int i = 0; i < n_problems; i++) {
//         std::shared_ptr<Problem2d> problem = problems[i];
//         int frame_id = frame_ids[i];
//         thread_pool.enqueue([problem, i, frame_id, &all_h, &all_h_dx, &all_h_dxdx, &all_phi1, &all_actuation, &all_lb, &all_ub,
//             &dq, &all_Jacobian, &all_postion, &all_theta, &all_dJdq, alpha0, gamma1, gamma2, compensation] {
//             try {
//                 xt::xtensor<double, 2> J = xt::view(all_Jacobian, frame_id, xt::keep(0,1,5), xt::all()); // shape (3, 7)
//                 xt::xtensor<double, 1> d = xt::view(all_postion, frame_id, xt::keep(0,1)); // shape (2, )
//                 double theta = all_theta(frame_id); // scalar
//                 xt::xtensor<double, 1> dJdq = xt::view(all_dJdq, frame_id, xt::keep(0,1,5)); // shape (3, )

//                 double alpha, h, dh, phi1, lb, ub;
//                 xt::xtensor<double, 1> alpha_dx, actuation, h_dx, v, dx;
//                 xt::xtensor<double, 2> alpha_dxdx, h_dxdx;
//                 std::tie(alpha, alpha_dx, alpha_dxdx) = problem->solve(d, theta);

//                 // Set gradients and hessians to zero for the orientation part
//                 alpha_dx(2) = 0;
//                 alpha_dxdx(0, 2) = 0;
//                 alpha_dxdx(1, 2) = 0;
//                 alpha_dxdx(2, 0) = 0;
//                 alpha_dxdx(2, 1) = 0;
//                 alpha_dxdx(2, 2) = 0;

//                 dx = xt::linalg::dot(J, dq); // shape (3)
                
//                 if (alpha != 0){
//                     h = alpha - alpha0;
//                     h_dx = alpha_dx;
//                     h_dxdx = alpha_dxdx;

//                     dh = xt::linalg::dot(h_dx, dx)(0);
//                     phi1 = dh + gamma1 * h;
//                     actuation = xt::linalg::dot(h_dx, J); // shape (7)
//                     lb = - gamma2 * phi1 - gamma1 * dh - xt::linalg::dot(xt::linalg::dot(dx, h_dxdx), dx)(0)
//                         - xt::linalg::dot(h_dx, dJdq)(0) + compensation;
//                     ub = std::numeric_limits<double>::infinity();
                    
//                 } else {
//                     h = std::numeric_limits<double>::infinity();
//                     h_dx = xt::zeros<double>({3});
//                     h_dxdx = xt::zeros<double>({3, 3});

//                     phi1 = 0;
//                     actuation = xt::zeros<double>({7});
//                     lb = 0;
//                     ub = 0;
//                 }

//                 all_h(i) = h;
//                 auto tmp_h_dx_view = xt::view(all_h_dx, i, xt::all());
//                 xt::noalias(tmp_h_dx_view) = h_dx;
//                 auto tmp_h_dxdx_view = xt::view(all_h_dxdx, i, xt::all(), xt::all());
//                 xt::noalias(tmp_h_dxdx_view) = h_dxdx;
//                 all_phi1(i) = phi1;
//                 auto tmp_actuation_view = xt::view(all_actuation, i, xt::all());
//                 xt::noalias(tmp_actuation_view) = actuation;
//                 all_lb(i) = lb;
//                 all_ub(i) = ub;

//             } catch (const std::exception& e) {
//                 std::cerr << "Exception in thread " << i << ": " << e.what() << std::endl;
//             }
//         });
//     }

//     thread_pool.wait();
    
//     return std::make_tuple(all_h, all_h_dx, all_h_dxdx, all_phi1, all_actuation, all_lb, all_ub);
// }
