#ifndef DNE_ORIGINAL_DNE_H
#define DNE_ORIGINAL_DNE_H

#include <iostream>
#include <Eigen/Sparse>
#include <Eigen/SparseCore>
#include <Eigen/SVD>
#include <chrono>
#include <ctime>
#include <gflags/gflags.h>

#include "params.h"
#include "../randomized_svd.h"
#include "../logging.h"

DEFINE_bool(original_svd_init, true, "Use Randomized SVD to init");

struct OriginalDNE{
  typedef std::unordered_map<size_t, size_t> TrainLabel;
  typedef Eigen::SparseMatrix<double, 0, std::ptrdiff_t> Sp;

  Sp const &S;
  TrainLabel  const &T;
  size_t const C;
  Params const params;

  explicit OriginalDNE(
    Sp const &S_,
    TrainLabel const &T_,
    size_t const C_,
    Params const params_):
  S(S_), T(T_), C(C_), params(params_){
  }

  void fit(Eigen::MatrixXd &W, Eigen::MatrixXd &B){
    srand(params.seed);

    report("Init method: " +
    std::string(FLAGS_original_svd_init ? "RandSVD" : "Random"));

    if(FLAGS_original_svd_init){
      Eigen::MatrixXd SC = S;
      auto svd = RandomizedSvd(SC, params.m);
      B = svd.matrixV().transpose();
      eq20(B, W);
    }else{
      W = Eigen::MatrixXd::Random(params.m, C);
      B = Eigen::MatrixXd::Random(params.m, S.rows());
    }

    for(size_t i = 1; i <= params.T_out; ++i){
      std::cout << "out: " << i << std::endl;
      eqB(W,B);
      eq20(B,W);
    }
  }

private:
  void WO(Eigen::MatrixXd const &W,
          Eigen::MatrixXd &outWO){
    assert(W.rows() == params.m and W.cols() == C);
    outWO = Eigen::MatrixXd::Zero(params.m, S.rows());

    Eigen::VectorXd sum_mc = W.rowwise().sum();
    for(auto &iter: T){
      auto i = iter.first;
      auto ci = iter.second;
      outWO.col(i) = sum_mc - (C * W.col(ci));
    }
  }

  void eq20(Eigen::MatrixXd const &B, Eigen::MatrixXd &outW){
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
      sgn(C * sum_1 - b_sum, w_c);
      outW.col(c) = w_c;
    }
  }

  void eqB(Eigen::MatrixXd const &W,
           Eigen::MatrixXd &B){
    Eigen::MatrixXd wo;
    WO(W, wo);
    sgn(-wo, B);
  }

  static void sgn(Eigen::MatrixXd const &x,
                  Eigen::MatrixXd &out){
    out = x.array().sign().matrix();
    out = (out.array() == 0).select(-1, out);
  }
};

#endif //DNE_ORIGINAL_DNE_H
