#ifndef DNE_RAW_SVD_DNE_H
#define DNE_RAW_SVD_DNE_H

#include <Eigen/Sparse>
#include <Eigen/SparseCore>
#include <Eigen/SVD>
#include "params.h"
#include "../binary.h"
#include "../randomized_svd.h"

struct RawSVD_DNE {
    typedef std::unordered_map<size_t, size_t> TrainLabel;
    typedef Eigen::SparseMatrix<double, 0, std::ptrdiff_t> Sp;

    Sp const &S;
    TrainLabel  const &T;
    size_t const C;
    Params const params;

    explicit RawSVD_DNE(
            Sp const &S_,
            TrainLabel const &T_,
            size_t const C_,
            Params const params_):
    S(S_), T(T_), C(C_), params(params_){
    }

    void fit(Eigen::MatrixXd &W, Eigen::MatrixXd &B){
      srand(params.seed);
      Eigen::MatrixXd SC = S;
      auto svd = RandomizedSvd(SC, params.m);
//      sgn( svd.matrixV().transpose(), B);
//      W = Eigen::MatrixXd::Zero(params.m, C);
//      eq13(B, W);
      W = Eigen::MatrixXd::Random(params.m, C);
      B = svd.matrixV().transpose();

      eq13(B, W);

      double loss_ = 0;
      for(size_t out = 1; out <= params.T_out; ++out){
        std::cout << "out: " << out << std::endl;

        for(size_t in = 1; in <= params.T_in; ++in){
          eq11(W, B);
        }
        eq13(B,W);

        if(params.check_loss){
           double loss_now = loss(W,B);
           std::cout << "loss: " << loss_now << std::endl;
           if(loss_now > loss_){
             std::cout << "loss increased !!!!" << std::endl;
           }

           loss_ = loss_now;
        }
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
    Eigen::MatrixXd B_Bt;
    if(B.coeff(1,1) == 1 or B.coeff(1,1) == -1){
      binary_mult512_self(B,B_Bt);
    }else{
      B_Bt = B * B.transpose();
    }

    Eigen::MatrixXd dLB = -B * S
      + params.lambda * wo;
      + params.mu * (B_Bt * B)
      + params.rho * (B * Eigen::VectorXd::Ones(N) * Eigen::RowVectorXd::Ones(N));

    Eigen::MatrixXd cf;
    CF(params.tau * B - dLB, B, cf);
    sgn(cf, B);
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
      sgn(C * sum_1 - b_sum, w_c);
      outW.col(c) = w_c;
    }
  }

  double loss(Eigen::MatrixXd const &W,
              Eigen::MatrixXd const &B){
    Eigen::MatrixXd wo;
    WO(W, wo);
    auto N = S.rows();

    return - 0.5 * (B * S * B.transpose()).trace()
           + params.lambda * (wo.transpose() * B).trace()
           + params.mu * 0.25 * (B * B.transpose()).trace()
           + params.rho * 0.5 * (B * Eigen::VectorXd::Zero(N)).trace();
  }

  static void sgn(Eigen::MatrixXd const &x,
                  Eigen::MatrixXd &out){
    out = x.array().sign().matrix();
    out = (out.array() == 0).select(-1, out);
  }

  static void CF(Eigen::MatrixXd const &x,
                 Eigen::MatrixXd const &y,
                 Eigen::MatrixXd &out){
    out = (x.array() == 0).select(y, x);
  }

};


#endif //DNE_RAW_DNE_H
