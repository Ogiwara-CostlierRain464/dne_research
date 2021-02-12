#ifndef DNE_REAL_DNE_H
#define DNE_REAL_DNE_H


#include <iostream>
#include <Eigen/Sparse>
#include <Eigen/SparseCore>

#include "params.h"

struct RealDNE{
  typedef std::unordered_map<size_t, size_t> TrainLabel;
  typedef Eigen::SparseMatrix<double, 0, std::ptrdiff_t> Sp;

  Sp const &S;
  TrainLabel  const &T;
  size_t const C;
  Params const params;

  explicit RealDNE(
    Sp const &S_,
    TrainLabel const &T_,
    size_t C_,
    Params const params_):
  S(S_), T(T_), C(C_), params(params_){
  }

  void fit(Eigen::MatrixXd &W, Eigen::MatrixXd &B){
    auto N = S.rows();
    srand(params.seed);
    W = Eigen::MatrixXd::Random(params.m, C);
    B = Eigen::MatrixXd::Random(params.m, N);

    for(size_t out = 1; out <= params.T_out; ++out){
      for(size_t in = 1; in <= params.T_in; ++in){
        eq11(W, B);
      }
      eq13(B, W);
    }
  }

  void WO(Eigen::MatrixXd const &W,
          Eigen::MatrixXd &outWO){
    assert(W.rows() == params.m and W.cols() == C);
    outWO = Eigen::MatrixXd::Zero(params.m, S.rows());

    Eigen::VectorXd sum_mc = W.rowwise().sum();
    for(auto &iter: T){
      auto i = iter.first;
      auto ci = iter.second;
      // W,Bに落とし込んだ時点で、もはやindexを保存していない
      // 前処理の段階で、インデックスの再割り当てを行おう
      outWO.col(i) = sum_mc - (C * W.col(ci));
    }
  }

  void eq11(Eigen::MatrixXd const &W,
            Eigen::MatrixXd &B){
    auto N = S.rows();
    assert(B.rows() == params.m and B.cols() == N);
    assert(W.rows() == params.m and W.cols() == C);
    Eigen::MatrixXd wo;
    WO(W, wo);
    Eigen::MatrixXd dLB = -B * S
                          + params.lambda * wo
                          + params.mu * (B * B.transpose() * B)
                          + params.rho * (B * Eigen::VectorXd::Ones(N) * Eigen::RowVectorXd::Ones(N));

    B = params.tau * B - dLB;
  }


  void eq13(Eigen::MatrixXd const &B, Eigen::MatrixXd &outW){
    assert(B.rows() == params.m and B.cols() == S.rows());

    outW = Eigen::MatrixXd::Zero(params.m, C);
    Eigen::VectorXd b_sum = Eigen::VectorXd::Zero(params.m);

    for(size_t i = 0; i < C; ++i){
      b_sum += B.col(i);
    }

    for(size_t c = 0; c < C; ++c){
      Eigen::VectorXd sum_1 = Eigen::VectorXd::Zero(params.m);
      for(auto &iter: T){
        auto i = iter.first;
        if(T.at(i) == c){
          sum_1 += B.col(i);
        }
      }

      Eigen::MatrixXd w_c;
      w_c = C * sum_1 - b_sum;
      outW.col(c) = w_c;
    }
  }
};

#endif //DNE_REAL_DNE_H
