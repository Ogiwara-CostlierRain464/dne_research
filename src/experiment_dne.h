#ifndef DNE_EXPERIMENT_DNE_H
#define DNE_EXPERIMENT_DNE_H

#include <iostream>
#include <Eigen/Sparse>
#include <Eigen/SparseCore>
#include <Eigen/SVD>
#include <chrono>
#include <ctime>
#include "randomized_svd.h"

class ExperimentDNE{
public:
  typedef std::unordered_map<size_t, size_t> TrainLabel;
  typedef Eigen::SparseMatrix<double, 0, std::ptrdiff_t> Sp;

  ExperimentDNE(Sp const &A_,
      TrainLabel const &T_,
      size_t N_, size_t M_, size_t C_,
      size_t L_, size_t T_in_):
  A(A_), T(T_), N(N_), M(M_),
  C(C_), L(L_), T_in(T_in_){
    assert(T.size() == L);
    std::cout << "ExperimentDNE" << std::endl;
  }

  /**
   * @param[out] W
   * @param[out] B
   */
  void fit(Eigen::MatrixXd &W, Eigen::MatrixXd &B){
    srand(time(nullptr));
    Sp S = (A + A * A) / 2;
    Eigen::MatrixXd SC = S;
    auto svd = RandomizedSvd(SC, M);

    std::cout << svd.matrixV().transpose() * svd.matrixV() << std::endl;

    sgn(svd.matrixU().transpose(), B);
    B = svd.singularValues().asDiagonal().toDenseMatrix().array().sqrt().matrix() * svd.matrixV().transpose();
//    sgn(svd.singularValues().asDiagonal().toDenseMatrix().array().sqrt().matrix() * svd.matrixV().transpose(),B);
//    sgn2((svd.matrixU() * (svd.singularValues().asDiagonal().toDenseMatrix().array().sqrt().matrix())).transpose(),B);
    W = Eigen::MatrixXd::Zero(M, C);
    eq13(B, W);

//    return;

//    W = Eigen::MatrixXd::Random(M, C);
//    B = Eigen::MatrixXd::Random(M, N);

    for(size_t _ = 1; _ <= 10; ++_){
      for(size_t i = 1; i <= T_in; ++i){
        eq11(W, B, S);
        std::cout << "updating B " << loss(B, S, W)  << std::endl;
      }
      eq13(B, W);
      std::cout << "updating W " << loss(B, S, W) << std::endl;
    }
  }

private:
  void WO(Eigen::MatrixXd const &W,
          Eigen::MatrixXd &outWO){
    assert(W.rows() == M and W.cols() == C);
    assert(T.size() == L);
    outWO = Eigen::MatrixXd::Zero(M, N);

    Eigen::VectorXd sum_mc = W.rowwise().sum();
//    Eigen::MatrixXd wo = Eigen::MatrixXd::Zero(M,N);
    for(auto &iter: T){
      auto i = iter.first;
      auto ci = T.at(i);
      outWO.col(i) = sum_mc - (C * W.col(ci));
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
            Sp const &S){
    assert(B.rows() == M and B.cols() == N);
    assert(S.rows() == N and S.cols() == N);
    assert(W.rows() == M and W.cols() == C);
    Eigen::MatrixXd wo;
    WO(W, wo);
    Eigen::MatrixXd dLB = -B * S
      + lambda * wo
      + mu * (B * B.transpose() * B)
      + rho * (B * Eigen::VectorXd::Ones(N) * Eigen::RowVectorXd::Ones(N));

//    Eigen::MatrixXd cf;
//    CF(tau * B - dLB, B, cf);
//    sgn2(cf, B);
    B = tau * B - dLB;
  }

  void eq13(Eigen::MatrixXd const &B, Eigen::MatrixXd &outW){
    assert(B.rows() == M and B.cols() == N);

    outW = Eigen::MatrixXd::Zero(M, C);
    Eigen::VectorXd b_sum = Eigen::VectorXd::Zero(M);

    for(size_t i = 0; i < C; ++i){
      b_sum += B.col(i);
    }

    for(size_t c = 0; c < C; ++c){
      Eigen::VectorXd sum_1 = Eigen::VectorXd::Zero(M);
      for(auto &iter: T){
        auto i = iter.first;
        if(T.at(i) == c){
          sum_1 += B.col(i);
        }
      }

      Eigen::MatrixXd w_c;
//      sgn(C * sum_1 - b_sum, w_c);
      w_c = C * sum_1 - b_sum;
      outW.col(c) = w_c;
    }
  }

  static void CF(Eigen::MatrixXd const &x,
                 Eigen::MatrixXd const &y,
                 Eigen::MatrixXd &out){
    out = (x.array()).select(y, x);
  }


  static void sgn(Eigen::MatrixXd const &x,
                  Eigen::MatrixXd &out){
    out = x.array().sign().matrix();
    out = (out.array() == 0).select(-1, out);
  }

  static void sgn2(Eigen::MatrixXd const &x,
                  Eigen::MatrixXd &out){
    double mean = x.array().mean();
    Eigen::MatrixXd signed_ = (x.array() > mean).select(+1, x);
    out = (signed_.array() <= mean).select(-1, signed_);
  }

  double loss(Eigen::MatrixXd const &B,
              Sp &S,
              Eigen::MatrixXd const &W){
    Eigen::MatrixXd wo;
    WO(W, wo);

    return - 0.5 * (B * S * B.transpose()).trace()
           + lambda * (wo.transpose() * B).trace()
           + mu * 0.25 * (B * B.transpose()).trace()
           + rho * 0.5 * (B * Eigen::VectorXd::Zero(N)).trace();
  }

  Sp const &A;
  TrainLabel const &T;

  size_t N;
  size_t M;
  size_t C;
  size_t L;
  size_t T_in;
  double tau = 1;
  double lambda = 0.00001;
  double mu = 0.00001;
  double rho = 0.00001;
};

#endif
