#include "ellipsoidAndLogSumExp3dPrb.hpp"

void EllipsoidAndLogSumExp3dPrb::validateProblem(int rob_frame_id, int obs_frame_id){

    if (SF_rob_ == nullptr){
        throw std::invalid_argument("SF_rob_ is nullptr!");
    } 

    if (SF_rob_->isMoving == false){
        throw std::invalid_argument("SF_rob_ is not moving!");
    }

    if (SF_rob_->isMoving == true and rob_frame_id < 0){
        throw std::invalid_argument("SF_rob_ is moving but given rob_frame_id < 0!");
    }

    if (SF_obs_ == nullptr){
        throw std::invalid_argument("SF_obs_ is nullptr!");
    }

    if (SF_obs_->isMoving == false && obs_frame_id >= 0){
        std::invalid_argument("SF_obs_ is not moving but given obs_frame_id >= 0!");
    }

    if (SF_obs_->isMoving == true && obs_frame_id < 0){
        std::invalid_argument("SF_obs_ is moving but given obs_frame_id < 0!");
    }
}

EllipsoidAndLogSumExp3dPrb::EllipsoidAndLogSumExp3dPrb(std::shared_ptr<Ellipsoid3d> SF1, 
    std::shared_ptr<LogSumExp3d> SF2, const xt::xtensor<double, 2>& obs_characteristic_points) :
    SF_rob_(SF1), SF_obs_(SF2), obs_characteristic_points_(obs_characteristic_points) {

    p_sol_ = xt::zeros<double>({dim_p_});
    xt::xtensor<double, 2> A_d = SF_obs_->A;
    xt::xtensor<double, 1> b_d = SF_obs_->b;
    double kappa_d = SF_obs_->kappa;
    n_exp_cone_ = A_d.shape()[0];
    m_ = 1 + 3*n_exp_cone_;
    n_ = dim_p_ + n_exp_cone_;
    double c_d = log(n_exp_cone_);

    A_d = kappa_d * A_d;
    b_d = kappa_d * b_d;

    A_scs_ = Eigen::SparseMatrix<double>(m_, n_);
    b_scs_ = Eigen::VectorXd::Zero(m_);

    // Positive cone
    for (int i=0; i<n_exp_cone_; ++i){
        A_scs_.insert(0, dim_p_+i) = 1;
    }
    b_scs_(0) = 1;

    // Exponential cones
    for (int i=0; i<n_exp_cone_; ++i){
        for (int j=0; j<dim_p_; ++j){
            A_scs_.insert(1 + i*3, j) = -A_d(i, j);
        }
        A_scs_.insert(1 + i*3 + 2, i + dim_p_) = -1;

        b_scs_(1 + i*3) = b_d(i) - c_d;
        b_scs_(1 + i*3 + 1) = 1;
    }

    double threshold = 1e-10;
    A_scs_.prune(threshold);

    scs_stgs_ = new ScsSettings();
    scs_set_default_settings(scs_stgs_);
    scs_stgs_->normalize = 1;
    scs_stgs_->eps_abs = 1e-5;
    scs_stgs_->eps_rel = 1e-5;
    scs_stgs_->verbose = 0;
    scs_stgs_->max_iters = 10000;
    scs_stgs_->time_limit_secs = 0.1;
    scs_stgs_->warm_start = 1;

    scs_cone_ = new ScsCone();
    scs_cone_->l = 1;
    scs_cone_->ep = n_exp_cone_;

    scs_info_ = new ScsInfo();

    scs_sol_ = new ScsSolution();
    scs_sol_->x = new double[n_]();
    scs_sol_->y = new double[m_]();
    scs_sol_->s = new double[m_]();
    warm_x_ = new double[n_]();
    warm_y_ = new double[m_]();
    warm_s_ = new double[m_]();
}

EllipsoidAndLogSumExp3dPrb::~EllipsoidAndLogSumExp3dPrb(){
    delete[] warm_x_;
    delete[] warm_y_;
    delete[] warm_s_;
    delete scs_stgs_;
    delete scs_info_;
    delete scs_cone_;
    delete[] scs_sol_->x;
    delete[] scs_sol_->y;
    delete[] scs_sol_->s;
    delete scs_sol_;
}

std::tuple<int, xt::xtensor<double, 1>> EllipsoidAndLogSumExp3dPrb::solveSCSPrb(const xt::xtensor<double, 1>& d, 
    const xt::xtensor<double, 1>& q){

    xt::xtensor<double, 2> Q_d = SF_rob_->getWorldQuadraticCoefficient(q);
    xt::xtensor<double, 1> mu_d = SF_rob_->getWorldCenter(d, q);

    Eigen::SparseMatrix<double> P(n_, n_);

    for (int i=0; i<dim_p_; ++i){
        for (int j=i; j<dim_p_; ++j){
            P.insert(i, j) = Q_d(i, j);
        }
    }
    double threshold = 1e-10;
    P.prune(threshold);

    ScsMatrix* P_scs = new ScsMatrix();
    P_scs->x = P.valuePtr();
    P_scs->i = P.innerIndexPtr();
    P_scs->p = P.outerIndexPtr();
    P_scs->m = P.rows();
    P_scs->n = P.cols();

    ScsMatrix* A_scs = new ScsMatrix();
    A_scs->x = A_scs_.valuePtr();
    A_scs->i = A_scs_.innerIndexPtr();
    A_scs->p = A_scs_.outerIndexPtr();
    A_scs->m = A_scs_.rows();
    A_scs->n = A_scs_.cols();

    Eigen::VectorXd c_scs = Eigen::VectorXd::Zero(n_);
    xt::xtensor<double, 1> c_d = - xt::linalg::dot(Q_d, mu_d);
    for (int i=0; i<dim_p_; ++i){
        c_scs(i) = c_d(i);
    }

    ScsData* scs_data = new ScsData();
    scs_data->m = m_;
    scs_data->n = n_;
    scs_data->A = A_scs;
    scs_data->P = P_scs;
    scs_data->b = b_scs_.data();
    scs_data->c = c_scs.data();

    ScsWork* scs_work = scs_init(scs_data, scs_cone_, scs_stgs_);

    // warm start: copy warm_x, warm_y, warm_s to scs_sol_
    for (int i=0; i<n_; ++i){
        scs_sol_->x[i] = warm_x_[i];
    }
    for (int i=0; i<m_; ++i){
        scs_sol_->y[i] = warm_y_[i];
        scs_sol_->s[i] = warm_s_[i];
    }

    int exitflag = scs_solve(scs_work, scs_sol_, scs_info_, 1);

    scs_finish(scs_work);

    if (exitflag == 1){
        xt::xtensor<double, 1> p_sol = xt::zeros<double>({dim_p_});
        for (int i=0; i<dim_p_; ++i){
            p_sol(i) = scs_sol_->x[i];
            p_sol_(i) = scs_sol_->x[i];
        }

        // warm start: copy scs_sol_ to warm_x, warm_y, warm_s
        for (int i=0; i<n_; ++i){
            warm_x_[i] = scs_sol_->x[i];
        }
        for (int i=0; i<m_; ++i){
            warm_y_[i] = scs_sol_->y[i];
            warm_s_[i] = scs_sol_->s[i];
        }

        delete P_scs;
        delete A_scs;
        delete scs_data;

        return std::make_tuple(exitflag, p_sol);
    } else {

        delete P_scs;
        delete A_scs;
        delete scs_data;

        return std::make_tuple(exitflag, xt::zeros<double>({dim_p_}));
    }
}

std::tuple<double, xt::xtensor<double, 1>, xt::xtensor<double, 2>> EllipsoidAndLogSumExp3dPrb::solve(
    const xt::xtensor<double, 1>& d, const xt::xtensor<double, 1>& q){
    
    // If the distance from d to each obs_characteristic_point is greater than 10 meters, return directly
    double min_dist = std::numeric_limits<double>::infinity();
    for (int i=0; i<obs_characteristic_points_.shape()[0]; ++i){
        double dist = xt::linalg::norm(xt::view(obs_characteristic_points_, i, xt::all()) - d);
        min_dist = std::min(min_dist, dist);
    }
    if (min_dist > 10){
        return std::make_tuple(0, xt::zeros<double>({7}), xt::zeros<double>({7, 7}));
    }
    
    int exitflag;
    xt::xtensor<double, 1> p;
    std::tie(exitflag, p) = solveSCSPrb(d, q);
    
    if (exitflag != 1){
        return std::make_tuple(0, xt::zeros<double>({7}), xt::zeros<double>({7, 7}));
    }

    double alpha;
    xt::xtensor<double, 1> alpha_dx;
    xt::xtensor<double, 2> alpha_dxdx;
    xt::xtensor<double, 1> d2 = xt::zeros<double>({3});
    xt::xtensor<double, 1> q2 = xt::zeros<double>({4});
    q2(3) = 1;

    std::tie(alpha, alpha_dx, alpha_dxdx) = getGradientAndHessian3d(p, SF_rob_, d, q, SF_obs_, d2, q2);

    return std::make_tuple(alpha, alpha_dx, alpha_dxdx);
}

std::tuple<double, xt::xtensor<double, 1>, xt::xtensor<double, 2>> EllipsoidAndLogSumExp3dPrb::solveMovingObstacle(
    const xt::xtensor<double, 1>& d1, const xt::xtensor<double, 1>& q1, const xt::xtensor<double, 1>& d2,
    const xt::xtensor<double, 1>& q2) {

    // If the distance from d to each obs_characteristic_point is greater than 10 meters, return directly
    double min_dist = std::numeric_limits<double>::infinity();
    xt::xtensor<double, 2> R = SF_obs_->getRotationMatrix(q2);
    for (int i=0; i<obs_characteristic_points_.shape()[0]; ++i){
        xt::xtensor<double, 1> tmp = xt::linalg::dot(R, xt::view(obs_characteristic_points_, i, xt::all())) + d2;
        double dist = xt::linalg::norm(tmp - d1);
        min_dist = std::min(min_dist, dist);
    }
    if (min_dist > 10){
        return std::make_tuple(0, xt::zeros<double>({14}), xt::zeros<double>({14, 14}));
    }
    
    // Update the exponential cone constraints
    xt::xtensor<double, 2> A_d = SF_obs_->getWorldMatrixCoefficient(q2);
    xt::xtensor<double, 1> b_d = SF_obs_->getWorldVectorCoefficient(d2, q2);
    double kappa_d = SF_obs_->kappa;
    n_exp_cone_ = A_d.shape()[0];
    m_ = 1 + 3*n_exp_cone_;
    n_ = dim_p_ + n_exp_cone_;
    double c_d = log(n_exp_cone_);

    A_d = kappa_d * A_d;
    b_d = kappa_d * b_d;

    A_scs_ = Eigen::SparseMatrix<double>(m_, n_);
    b_scs_ = Eigen::VectorXd::Zero(m_);

    // Positive cone
    for (int i=0; i<n_exp_cone_; ++i){
        A_scs_.insert(0, dim_p_+i) = 1;
    }
    b_scs_(0) = 1;

    // Exponential cones
    for (int i=0; i<n_exp_cone_; ++i){
        for (int j=0; j<dim_p_; ++j){
            A_scs_.insert(1 + i*3, j) = -A_d(i, j);
        }
        A_scs_.insert(1 + i*3 + 2, i + dim_p_) = -1;

        b_scs_(1 + i*3) = b_d(i) - c_d;
        b_scs_(1 + i*3 + 1) = 1;
    }

    double threshold = 1e-10;
    A_scs_.prune(threshold);

    int exitflag;
    xt::xtensor<double, 1> p;
    std::tie(exitflag, p) = solveSCSPrb(d1, q1);
    
    if (exitflag != 1){
        return std::make_tuple(0, xt::zeros<double>({14}), xt::zeros<double>({14, 14}));
    }

    double alpha;
    xt::xtensor<double, 1> alpha_dx;
    xt::xtensor<double, 2> alpha_dxdx;

    std::tie(alpha, alpha_dx, alpha_dxdx) = getGradientAndHessian3d(p, SF_rob_, d1, q1, SF_obs_, d2, q2);

    return std::make_tuple(alpha, alpha_dx, alpha_dxdx);
}
