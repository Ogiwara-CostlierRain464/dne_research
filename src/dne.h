#ifndef DNE_DNE_H
#define DNE_DNE_H

#include <iostream>
#include <Eigen/Sparse>
#include <Eigen/SparseCore>


class DNE{
public:
  typedef std::vector<std::vector<size_t>> TrainLabel;

  DNE(Eigen::MatrixXd const &A_,
      TrainLabel const &T_,
      size_t N_, size_t M_, size_t C_,
      size_t L_, size_t T_in_):
  A(A_), T(T_), N(N_), M(M_),
  C(C_), L(L_), T_in(T_in_){
    // wanna check A is symmetric.
  }

  /**
   * @param[out] W
   * @param[out] B
   */
  void fit(Eigen::MatrixXd &W, Eigen::MatrixXd &B){
    srand(time(nullptr));
    Eigen::MatrixXd S = (A + A * A) / 2;
    W = Eigen::MatrixXd::Random(M, C);
    B = Eigen::MatrixXd::Random(M, N);

    for(size_t _ = 1; _ <= 20; ++_){
      for(size_t i = 1; i <= T_in; ++i){
        eq11(W, B, S);
        std::cout << "updating B" << loss(B, S, W) << std::endl;
      }
      eq13(B, W);
      std::cout << "updating W" << loss(B, S, W) << std::endl;
    }
  }

private:
  void WO(Eigen::MatrixXd const &W,
          Eigen::MatrixXd &outWO){
    assert(W.rows() == M and W.cols() == C);
    assert(T.size() == L);
    outWO = Eigen::MatrixXd::Zero(M, N);

    Eigen::VectorXd sum_mc = W.rowwise().sum();
    for(size_t i = 0; i < L ; ++i){
      Eigen::VectorXd wo_i = Eigen::VectorXd::Zero(M);
      for(auto j : T[i]){
        wo_i += sum_mc - (C * W.col(j));
      }
      outWO.col(i) = wo_i;
    }
  }

  /**
   *
   * @param W
   * @param[in,out] B
   * @param S
   */
  void eq11(Eigen::MatrixXd const &W,
            Eigen::MatrixXd &B,
            Eigen::MatrixXd const &S){
    assert(B.rows() == M and B.cols() == N);
    assert(S.rows() == N and S.cols() == N);
    assert(W.rows() == M and W.cols() == C);
    Eigen::MatrixXd wo;
    WO(W, wo);
    Eigen::MatrixXd dLB = -B * S
      + lambda * wo
      + mu * (B * B.transpose() * B)
      + rho * (B * Eigen::VectorXd::Ones(N) * Eigen::RowVectorXd::Ones(N));

    Eigen::MatrixXd cf;
    CF(tau * B - dLB, B, cf);
    sgn(cf, B);
  }

  void eq13(Eigen::MatrixXd const &B, Eigen::MatrixXd &outW){
    assert(B.rows() == M and B.cols() == N);

    outW = Eigen::MatrixXd::Zero(M, C);

    for(size_t c = 0; c < C; ++c){
      Eigen::VectorXd sum_1 = Eigen::VectorXd::Zero(M);
      Eigen::VectorXd sum_2 = Eigen::VectorXd::Zero(M);
      for(size_t i = 0; i < L; ++i){
        if(std::count(T[i].begin(), T[i].end(), c))
          sum_1 += B.col(i);

      }
      for(size_t i = 0; i < L; ++i){
        auto n_ki = T[i].size();
        sum_2 += n_ki * B.col(i);
      }

      Eigen::MatrixXd w_c;
      sgn(C * sum_1 - sum_2, w_c);
      outW.col(c) = w_c;
    }
  }

  static void CF(Eigen::MatrixXd const &x,
                 Eigen::MatrixXd const &y,
                 Eigen::MatrixXd &out){
    out = (x.array() == 0).select(y, x);
  }

  static void sgn(Eigen::MatrixXd const &x,
                  Eigen::MatrixXd &out){
    out = x.array().sign().matrix();
    out = (out.array() == 0).select(-1, out);
  }

  double loss(Eigen::MatrixXd const &B,
              Eigen::MatrixXd const &S,
              Eigen::MatrixXd const &W){
    Eigen::MatrixXd wo;
    WO(W, wo);

    return - 0.5 * (B * S * B.transpose()).trace()
           + lambda * (wo.transpose() * B).trace()
           + mu * 0.25 * (B * B.transpose()).trace()
           + rho * 0.5 * (B * Eigen::VectorXd::Zero(N)).trace();
  }

  Eigen::MatrixXd const &A;
  TrainLabel const &T;

  size_t N;
  size_t M;
  size_t C;
  size_t L;
  size_t T_in;
  double tau = 5;
  double lambda = 0.5;
  double mu = 0.01;
  double rho = 0.01;
};

#endif //DNE_DNE_H