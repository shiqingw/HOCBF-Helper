#include "problem3dCollectionMovingObstacle.hpp"

Problem3dCollectionMovingObstacle::Problem3dCollectionMovingObstacle(size_t n_threads_) : thread_pool(n_threads_) {
    n_threads = n_threads_;
}

Problem3dCollectionMovingObstacle::~Problem3dCollectionMovingObstacle() {
    thread_pool.wait();
}

void Problem3dCollectionMovingObstacle::addProblem(std::shared_ptr<Problem3d> problem, int rob_frame_id, int obs_frame_id) {
    problem->validateProblem(rob_frame_id, obs_frame_id);
    problems.push_back(problem);
    rob_frame_ids.push_back(rob_frame_id);
    obs_frame_ids.push_back(obs_frame_id);
    n_problems++;
}

void Problem3dCollectionMovingObstacle::waitAll() {
    thread_pool.wait();
}

void Problem3dCollectionMovingObstacle::stopAll() {
    thread_pool.stopAll();
}

std::tuple<xt::xtensor<double, 1>, xt::xtensor<double, 2>, xt::xtensor<double, 3>> Problem3dCollectionMovingObstacle::solveGradientAndHessian(
        const xt::xtensor<double, 2>& all_d_rob, const xt::xtensor<double, 2>& all_q_rob,
        const xt::xtensor<double, 2>& all_d_obs, const xt::xtensor<double, 2>& all_q_obs) {

    xt::xtensor<double, 1> all_alpha = xt::zeros<double>({n_problems});
    xt::xtensor<double, 2> all_alpha_dx = xt::zeros<double>({n_problems, 14});
    xt::xtensor<double, 3> all_alpha_dxdx = xt::zeros<double>({n_problems, 14, 14});

    for (int i = 0; i < n_problems; i++) {
        std::shared_ptr<Problem3d> problem = problems[i];
        int rob_frame_id = rob_frame_ids[i];
        int obs_frame_id = obs_frame_ids[i];
        thread_pool.enqueue([i, rob_frame_id, obs_frame_id, problem, &all_alpha, &all_alpha_dx, &all_alpha_dxdx,
            &all_d_rob, &all_q_rob, &all_d_obs, &all_q_obs] {
            try {
                xt::xtensor<double, 1> d1 = xt::view(all_d_rob, rob_frame_id, xt::all()); // shape (3)
                xt::xtensor<double, 1> q1 = xt::view(all_q_rob, rob_frame_id, xt::all()); // shape (7)

                xt::xtensor<double, 1> d2 = xt::view(all_d_obs, obs_frame_id, xt::all()); // shape (3)
                xt::xtensor<double, 1> q2 = xt::view(all_q_obs, obs_frame_id, xt::all()); // shape (7)

                double alpha;
                xt::xtensor<double, 1> alpha_dx;
                xt::xtensor<double, 2> alpha_dxdx;
                std::tie(alpha, alpha_dx, alpha_dxdx) = problem->solveMovingObstacle(d1, q1, d2, q2);
                
                all_alpha(i) = alpha;
                auto tmp_all_alpha_dx_view = xt::view(all_alpha_dx, i, xt::all());
                xt::noalias(tmp_all_alpha_dx_view) = alpha_dx;
                auto tmp_all_alpha_dxdx_view = xt::view(all_alpha_dxdx, i, xt::all(), xt::all());
                xt::noalias(tmp_all_alpha_dxdx_view) = alpha_dxdx;
                
            } catch (const std::exception& e) {
                std::cerr << "Exception in thread " << i << ": " << e.what() << std::endl;
            }
        });
    }

    thread_pool.wait();

    return std::make_tuple(all_alpha, all_alpha_dx, all_alpha_dxdx);
}

std::tuple<xt::xtensor<double, 1>, xt::xtensor<double, 1>, xt::xtensor<double, 2>, xt::xtensor<double, 1>, 
    xt::xtensor<double, 1>> Problem3dCollectionMovingObstacle::getCBFConstraints(
    const xt::xtensor<double, 1>& dq_rob, const xt::xtensor<double, 2>& all_position_rob, const xt::xtensor<double, 2>& all_quat_rob, 
    const xt::xtensor<double, 3>& all_Jacobian_rob, const xt::xtensor<double, 2>& all_dJdq_rob,
    const xt::xtensor<double, 2>& all_position_obs, const xt::xtensor<double, 2>& all_quat_obs,
    const xt::xtensor<double, 2>& all_velocity_obs, const xt::xtensor<double, 2>& all_omega_obs,
    const xt::xtensor<double, 2>& all_velocity_dot_obs, const xt::xtensor<double, 2>& all_omega_dot_obs,
    double alpha0, double gamma1, double gamma2, double compensation){

    xt::xtensor<double, 1> all_h = xt::zeros<double>({n_problems});
    xt::xtensor<double, 1> all_phi1 = xt::zeros<double>({n_problems});
    xt::xtensor<double, 2> all_actuation = xt::zeros<double>({n_problems, 7});
    xt::xtensor<double, 1> all_lb = xt::zeros<double>({n_problems});
    xt::xtensor<double, 1> all_ub = xt::zeros<double>({n_problems});

    for (int i = 0; i < n_problems; i++) {
        std::shared_ptr<Problem3d> problem = problems[i];
        int rob_frame_id = rob_frame_ids[i];
        int obs_frame_id = obs_frame_ids[i];
        thread_pool.enqueue([problem, i, rob_frame_id, obs_frame_id, &all_h, &all_phi1, &all_actuation, &all_lb, &all_ub,
            &dq_rob, &all_Jacobian_rob, &all_position_rob, &all_quat_rob, &all_dJdq_rob, &all_position_obs, &all_quat_obs,
            &all_velocity_obs, &all_omega_obs, &all_velocity_dot_obs, &all_omega_dot_obs,
            alpha0, gamma1, gamma2, compensation] {
            try {
                if (obs_frame_id < 0){// static obstacle
                    xt::xtensor<double, 2> J_rob = xt::view(all_Jacobian_rob, rob_frame_id, xt::all(), xt::all()); // shape (6, 7)
                    xt::xtensor<double, 1> d_rob = xt::view(all_position_rob, rob_frame_id, xt::all()); // shape (3, )
                    xt::xtensor<double, 1> q_rob = xt::view(all_quat_rob, rob_frame_id, xt::all()); // shape (4, )
                    xt::xtensor<double, 1> dJdq_rob = xt::view(all_dJdq_rob, rob_frame_id, xt::all()); // shape (6, )
                    
                    double alpha, h, dh, phi1, lb, ub;
                    xt::xtensor<double, 1> alpha_dx, actuation, h_dx, v_omega_rob, dx, dquat_rob;
                    xt::xtensor<double, 2> alpha_dxdx, h_dxdx, A_rob, Q_rob, dA_rob, dQ_rob;
                    std::tie(alpha, alpha_dx, alpha_dxdx) = problem->solve(d_rob, q_rob);
                    
                    v_omega_rob = xt::linalg::dot(J_rob, dq_rob); // shape (6)
                    Q_rob = getQMatrixFromQuat(q_rob); // shape (4, 3)
                    A_rob = xt::zeros<double>({7, 6});
                    auto tmp_A11_rob_view = xt::view(A_rob, xt::range(0, 3), xt::range(0, 3));
                    xt::noalias(tmp_A11_rob_view) = xt::eye<double>(3);

                    auto tmp_A22_rob_view = xt::view(A_rob, xt::range(3, 7), xt::range(3, 6));
                    xt::noalias(tmp_A22_rob_view) = 0.5*Q_rob;

                    dx = xt::linalg::dot(A_rob, v_omega_rob); // shape (7)
                    dquat_rob = 0.5 * xt::linalg::dot(Q_rob, xt::view(v_omega_rob, xt::range(3, 6))); // shape (4)
                    dQ_rob = getdQMatrixFromdQuat(dquat_rob); // shape (4, 3)
                    dA_rob = xt::zeros<double>({7, 6});
                    auto tmp_dA22_rob_view = xt::view(dA_rob, xt::range(3, 7), xt::range(3, 6));
                    xt::noalias(tmp_dA22_rob_view) = 0.5*dQ_rob;

                    if (alpha != 0){
                        h = alpha - alpha0;
                        h_dx = alpha_dx;
                        h_dxdx = alpha_dxdx;

                        dh = xt::linalg::dot(h_dx, dx)(0);
                        phi1 = dh + gamma1 * h;
                        actuation = xt::linalg::dot(xt::linalg::dot(h_dx, A_rob), J_rob); // shape (7)
                        lb = - gamma2 * phi1 - gamma1 * dh - xt::linalg::dot(xt::linalg::dot(dx, h_dxdx), dx)(0)
                            - xt::linalg::dot(h_dx, xt::linalg::dot(dA_rob, v_omega_rob))(0) - xt::linalg::dot(h_dx, xt::linalg::dot(A_rob, dJdq_rob))(0)
                            + compensation;
                        ub = std::numeric_limits<double>::infinity();
                    } else {
                        h = std::numeric_limits<double>::infinity();

                        phi1 = 0;
                        actuation = xt::zeros<double>({7});
                        lb = 0;
                        ub = 0;
                    }
                    all_h(i) = h;
                    all_phi1(i) = phi1;
                    auto tmp_actuation_view = xt::view(all_actuation, i, xt::all());
                    xt::noalias(tmp_actuation_view) = actuation;
                    all_lb(i) = lb;
                    all_ub(i) = ub;

                } else {
                    xt::xtensor<double, 2> J_rob = xt::view(all_Jacobian_rob, rob_frame_id, xt::all(), xt::all()); // shape (6, 7)
                    xt::xtensor<double, 1> d_rob = xt::view(all_position_rob, rob_frame_id, xt::all()); // shape (3, )
                    xt::xtensor<double, 1> q_rob = xt::view(all_quat_rob, rob_frame_id, xt::all()); // shape (4, )
                    xt::xtensor<double, 1> dJdq_rob = xt::view(all_dJdq_rob, rob_frame_id, xt::all()); // shape (6, )
                    
                    xt::xtensor<double, 1> d_obs = xt::view(all_position_obs, obs_frame_id, xt::all()); // shape (3, )
                    xt::xtensor<double, 1> q_obs = xt::view(all_quat_obs, obs_frame_id, xt::all()); // shape (4, )
                    xt::xtensor<double, 1> v_obs = xt::view(all_velocity_obs, obs_frame_id, xt::all()); // shape (3, )
                    xt::xtensor<double, 1> omega_obs = xt::view(all_omega_obs, obs_frame_id, xt::all()); // shape (3, )
                    xt::xtensor<double, 1> dv_obs = xt::view(all_velocity_dot_obs, obs_frame_id, xt::all()); // shape (3, )
                    xt::xtensor<double, 1> domega_obs = xt::view(all_omega_dot_obs, obs_frame_id, xt::all()); // shape (3, )

                    double alpha, h, dh, phi1, lb, ub;
                    xt::xtensor<double, 1> alpha_dx, actuation, h_dx, v_omega_rob, dx, dquat_rob, dquat_obs, ddquat_obs;
                    xt::xtensor<double, 2> alpha_dxdx, h_dxdx, A_rob, Q_rob, dA_rob, dQ_rob, Q_obs, dQ_obs;
                    std::tie(alpha, alpha_dx, alpha_dxdx) = problem->solveMovingObstacle(d_rob, q_rob, d_obs, q_obs);

                    v_omega_rob = xt::linalg::dot(J_rob, dq_rob); // shape (6)
                    Q_rob = getQMatrixFromQuat(q_rob); // shape (4, 3)
                    A_rob = xt::zeros<double>({7, 6});
                    auto tmp_A11_rob_view = xt::view(A_rob, xt::range(0, 3), xt::range(0, 3));
                    xt::noalias(tmp_A11_rob_view) = xt::eye<double>(3);
                    auto tmp_A22_rob_view = xt::view(A_rob, xt::range(3, 7), xt::range(3, 6));
                    xt::noalias(tmp_A22_rob_view) = 0.5*Q_rob;

                    Q_obs = getQMatrixFromQuat(q_obs); // shape (4, 3)
                    dquat_obs = 0.5 * xt::linalg::dot(Q_obs, omega_obs); // shape (4)

                    dx = xt::zeros<double>({14}); //[v_rob, dquat_rob, v_obs, dquat_obs]
                    auto dx_rob_view = xt::view(dx, xt::range(0, 7));
                    xt::noalias(dx_rob_view) = xt::linalg::dot(A_rob, v_omega_rob); // shape (7)
                    auto dx_obs_v_view = xt::view(dx, xt::range(7, 10));
                    xt::noalias(dx_obs_v_view) = v_obs;
                    auto dx_obs_dquat_view = xt::view(dx, xt::range(10, 14));
                    xt::noalias(dx_obs_dquat_view) = dquat_obs;

                    dquat_rob = 0.5 * xt::linalg::dot(Q_rob, xt::view(v_omega_rob, xt::range(3, 6))); // shape (4)
                    dQ_rob = getdQMatrixFromdQuat(dquat_rob); // shape (4, 3)
                    dA_rob = xt::zeros<double>({7, 6});
                    auto tmp_dA22_rob_view = xt::view(dA_rob, xt::range(3, 7), xt::range(3, 6));
                    xt::noalias(tmp_dA22_rob_view) = 0.5*dQ_rob;

                    dQ_obs = getdQMatrixFromdQuat(dquat_obs); // shape (4, 3)
                    ddquat_obs = 0.5 * xt::linalg::dot(dQ_obs, omega_obs) + 0.5 * xt::linalg::dot(Q_obs, domega_obs); // shape (4)

                    if (alpha != 0){
                        h = alpha - alpha0;
                        h_dx = alpha_dx;
                        h_dxdx = alpha_dxdx;

                        dh = xt::linalg::dot(h_dx, dx)(0);
                        phi1 = dh + gamma1 * h;
                        actuation = xt::linalg::dot(xt::linalg::dot(xt::view(h_dx, xt::range(0, 7)), A_rob), J_rob); // shape (7)
                        xt::xtensor<double, 1> ddx_rest = xt::zeros<double>({14});
                        auto ddx_rest_rob_view = xt::view(ddx_rest, xt::range(0, 7));
                        xt::noalias(ddx_rest_rob_view) = xt::linalg::dot(dA_rob, v_omega_rob) + xt::linalg::dot(A_rob, dJdq_rob);
                        auto ddx_rest_obs_v_view = xt::view(ddx_rest, xt::range(7, 10));
                        xt::noalias(ddx_rest_obs_v_view) = dv_obs;
                        auto ddx_rest_obs_dquat_view = xt::view(ddx_rest, xt::range(10, 14));
                        xt::noalias(ddx_rest_obs_dquat_view) = ddquat_obs;
                        lb = - gamma2 * phi1 - gamma1 * dh - xt::linalg::dot(xt::linalg::dot(dx, h_dxdx), dx)(0)
                            - xt::linalg::dot(h_dx, ddx_rest)(0) + compensation;
                        ub = std::numeric_limits<double>::infinity();
                    } else {
                        h = std::numeric_limits<double>::infinity();
                        phi1 = 0;
                        actuation = xt::zeros<double>({7});
                        lb = 0;
                        ub = 0;
                    }
                    all_h(i) = h;
                    all_phi1(i) = phi1;
                    auto tmp_actuation_view = xt::view(all_actuation, i, xt::all());
                    xt::noalias(tmp_actuation_view) = actuation;
                    all_lb(i) = lb;
                    all_ub(i) = ub;
                }

            } catch (const std::exception& e) {
                std::cerr << "Exception in thread " << i << ": " << e.what() << std::endl;
            }
        });
    }

    thread_pool.wait();
    
    return std::make_tuple(all_h, all_phi1, all_actuation, all_lb, all_ub);
}

// std::tuple<xt::xtensor<double, 1>, xt::xtensor<double, 2>, xt::xtensor<double, 3>, xt::xtensor<double, 1>, 
//     xt::xtensor<double, 1>, xt::xtensor<double, 2>> Problem3dCollection::getSmoothMinCBFConstraints(
//     const xt::xtensor<double, 1>& dq, const xt::xtensor<double, 2>& all_postion, const xt::xtensor<double, 2>& all_quat, 
//     const xt::xtensor<double, 3>& all_Jacobian, const xt::xtensor<double, 2>& all_dJdq, double alpha0){

//     xt::xtensor<double, 1> all_h = xt::zeros<double>({n_problems});
//     xt::xtensor<double, 2> all_h_dx = xt::zeros<double>({n_problems, 7});
//     xt::xtensor<double, 3> all_h_dxdx = xt::zeros<double>({n_problems, 7, 7});
//     xt::xtensor<double, 1> all_first_order_average_scalar = xt::zeros<double>({n_problems});
//     xt::xtensor<double, 1> all_second_order_average_scalar = xt::zeros<double>({n_problems});
//     xt::xtensor<double, 2> all_second_order_average_vector = xt::zeros<double>({n_problems, 7});

//     for (int i = 0; i < n_problems; i++) {
//         std::shared_ptr<Problem3d> problem = problems[i];
//         int frame_id = frame_ids[i];
//         thread_pool.enqueue([problem, i, frame_id, &all_h, &all_h_dx, &all_h_dxdx, &all_first_order_average_scalar, &all_second_order_average_scalar,
//             &all_second_order_average_vector, &dq, &all_Jacobian, &all_postion, &all_quat, &all_dJdq, alpha0] {
//             try {
//                 xt::xtensor<double, 2> J = xt::view(all_Jacobian, frame_id, xt::all(), xt::all()); // shape (6, 7)
//                 xt::xtensor<double, 1> d = xt::view(all_postion, frame_id, xt::all()); // shape (3, )
//                 xt::xtensor<double, 1> q = xt::view(all_quat, frame_id, xt::all()); // shape (4, )
//                 xt::xtensor<double, 1> dJdq = xt::view(all_dJdq, frame_id, xt::all()); // shape (6, )

//                 double alpha, h, first_order_average_scalar, second_order_average_scalar;
//                 xt::xtensor<double, 1> alpha_dx, h_dx, v, dx, dquat, second_order_average_vector;
//                 xt::xtensor<double, 2> alpha_dxdx, h_dxdx, A, Q, dA, dQ;
//                 std::tie(alpha, alpha_dx, alpha_dxdx) = problem->solve(d, q);
                
//                 v = xt::linalg::dot(J, dq); // shape (6)
//                 Q = getQMatrixFromQuat(q); // shape (4, 3)
//                 A = xt::zeros<double>({7, 6});
//                 auto tmp_A11_view = xt::view(A, xt::range(0, 3), xt::range(0, 3));
//                 xt::noalias(tmp_A11_view) = xt::eye<double>(3);

//                 auto tmp_A22_view = xt::view(A, xt::range(3, 7), xt::range(3, 6));
//                 xt::noalias(tmp_A22_view) = 0.5*Q;

//                 dx = xt::linalg::dot(A, v); // shape (7)
//                 dquat = 0.5 * xt::linalg::dot(Q, xt::view(v, xt::range(3, 6))); // shape (4)
//                 dQ = getdQMatrixFromdQuat(dquat); // shape (4, 3)
//                 dA = xt::zeros<double>({7, 6});
//                 auto tmp_dA22_view = xt::view(dA, xt::range(3, 7), xt::range(3, 6));
//                 xt::noalias(tmp_dA22_view) = 0.5*dQ;

//                 if (alpha != 0){
//                     h = alpha - alpha0;
//                     h_dx = alpha_dx;
//                     h_dxdx = alpha_dxdx;

//                     first_order_average_scalar = xt::linalg::dot(h_dx, dx)(0);

//                     second_order_average_scalar = xt::linalg::dot(xt::linalg::dot(dx, h_dxdx), dx)(0)
//                         + xt::linalg::dot(h_dx, xt::linalg::dot(dA, v))(0) + xt::linalg::dot(h_dx, xt::linalg::dot(A, dJdq))(0);

//                     second_order_average_vector = xt::linalg::dot(xt::linalg::dot(h_dx, A), J); // shape (7)
//                 } else {
//                     h = std::numeric_limits<double>::infinity();
//                     h_dx = xt::zeros<double>({7});
//                     h_dxdx = xt::zeros<double>({7, 7});
                    
//                     first_order_average_scalar = 0;
//                     second_order_average_scalar = 0;
//                     second_order_average_vector = xt::zeros<double>({7});
//                 }

//                 all_h(i) = h;
//                 auto tmp_h_dx_view = xt::view(all_h_dx, i, xt::all());
//                 xt::noalias(tmp_h_dx_view) = h_dx;
//                 auto tmp_h_dxdx_view = xt::view(all_h_dxdx, i, xt::all(), xt::all());
//                 xt::noalias(tmp_h_dxdx_view) = h_dxdx;
//                 all_first_order_average_scalar(i) = first_order_average_scalar;
//                 all_second_order_average_scalar(i) = second_order_average_scalar;
//                 auto tmp_second_order_average_vector_view = xt::view(all_second_order_average_vector, i, xt::all());
//                 xt::noalias(tmp_second_order_average_vector_view) = second_order_average_vector;

//             } catch (const std::exception& e) {
//                 std::cerr << "Exception in thread " << i << ": " << e.what() << std::endl;
//             }
//         });
//     }

//     thread_pool.wait();
    
//     return std::make_tuple(all_h, all_h_dx, all_h_dxdx, all_first_order_average_scalar, all_second_order_average_scalar, all_second_order_average_vector);
// }