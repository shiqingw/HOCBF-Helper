#include "elliposoidAndLogSumExp3dPrb.hpp"

ElliposoidAndLogSumExp3dPrb::ElliposoidAndLogSumExp3dPrb(std::shared_ptr<Ellipsoid3d> SF1, 
    std::shared_ptr<LogSumExp3d> SF2) : SF_rob_(SF1), SF_obs_(SF2) {

        p_sol_ = xt::zeros<double>({dim_p_});
        xt::xarray<double> A_d = SF_obs_->A;
        xt::xarray<double> b_d = SF_obs_->b;
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
            A_scs_.insert(1 + i*3, 0) = -A_d(i, 0);
            A_scs_.insert(1 + i*3, 1) = -A_d(i, 1);
            A_scs_.insert(1 + i*3, 2) = -A_d(i, 2);
            A_scs_.insert(1 + i*3 + 2, i + dim_p_) = -1;

            b_scs_(1 + i*3) = b_d(i) - c_d;
            b_scs_(1 + i*3 + 1) = 1;
        }

        double threshold = 1e-10;
        A_scs_.prune(threshold);

        scs_stgs_ = new ScsSettings();
        scs_set_default_settings(scs_stgs_);
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

ElliposoidAndLogSumExp3dPrb::~ElliposoidAndLogSumExp3dPrb(){
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

std::tuple<int, xt::xarray<double>> ElliposoidAndLogSumExp3dPrb::solve_scs_prb(const xt::xarray<double>& d, 
    const xt::xarray<double>& q){

    xt::xarray<double> Q_d = SF_rob_->getWorldQuadraticCoefficient(q);
    xt::xarray<double> mu_d = SF_rob_->getWorldCenter(d, q);

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
    xt::xarray<double> c_d = - xt::linalg::dot(Q_d, mu_d);
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
        xt::xarray<double> p_sol = xt::zeros<double>({dim_p_});
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

std::tuple<double, xt::xarray<double>, xt::xarray<double>> ElliposoidAndLogSumExp3dPrb::solve(
    const xt::xarray<double>& d, const xt::xarray<double>& q){
    int exitflag;
    xt::xarray<double> p;
    std::tie(exitflag, p) = solve_scs_prb(d, q);
    
    if (exitflag != 1){
        return std::make_tuple(0, xt::zeros<double>({7}), xt::zeros<double>({7, 7}));
    }

    double alpha;
    xt::xarray<double> alpha_dx;
    xt::xarray<double> alpha_dxdx;
    xt::xarray<double> d2 = xt::zeros<double>({3});
    xt::xarray<double> q2 = xt::zeros<double>({4});
    q2(3) = 1;

    std::tie(alpha, alpha_dx, alpha_dxdx) = getGradientAndHessian3d(p, SF_rob_, d, q, SF_obs_, d2, q2);

    return std::make_tuple(alpha, alpha_dx, alpha_dxdx);
}

// std::tuple<double, xt::xarray<double>, xt::xarray<double>> ElliposoidAndLogSumExp3dPrb::solve(
//     const xt::xarray<double>& d, const xt::xarray<double>& q){
    
//     auto start_total = std::chrono::high_resolution_clock::now();
    
//     auto start_scs = std::chrono::high_resolution_clock::now();
//     xt::xarray<double> p = solve_scs_prb(d, q);
//     auto end_scs = std::chrono::high_resolution_clock::now();
//     std::chrono::duration<double> diff_scs = end_scs - start_scs;
//     std::cout << "solve_scs_prb took: " << diff_scs.count() << " s\n";
    
//     auto start_init = std::chrono::high_resolution_clock::now();
//     double alpha;
//     xt::xarray<double> alpha_dx;
//     xt::xarray<double> alpha_dxdx;
//     xt::xarray<double> d2 = xt::zeros<double>({3});
//     xt::xarray<double> q2 = xt::zeros<double>({4});
//     q2(3) = 1;
//     auto end_init = std::chrono::high_resolution_clock::now();
//     std::chrono::duration<double> diff_init = end_init - start_init;
//     std::cout << "Initialization took: " << diff_init.count() << " s\n";

//     auto start_gh = std::chrono::high_resolution_clock::now();
//     std::tie(alpha, alpha_dx, alpha_dxdx) = getGradientAndHessian3d(p, SF_rob_, d, q, SF_obs_, d2, q2);
//     auto end_gh = std::chrono::high_resolution_clock::now();
//     std::chrono::duration<double> diff_gh = end_gh - start_gh;
//     std::cout << "getGradientAndHessian3d took: " << diff_gh.count() << " s\n";
    
//     auto end_total = std::chrono::high_resolution_clock::now();
//     std::chrono::duration<double> diff_total = end_total - start_total;
//     std::cout << "Total solve function took: " << diff_total.count() << " s\n";

//     return std::make_tuple(alpha, alpha_dx, alpha_dxdx);
// }
