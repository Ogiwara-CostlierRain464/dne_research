#include <boost/serialization/unordered_map.hpp>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <Eigen/Sparse>
#include "dataset_repo.h"
#include "model/raw_dne.h"
#include "model/raw_svd_dne.h"
#include "model/original_dne.h"
#include "model/real_dne.h"
#include "model/raw_semi_dne.h"
#include "logging.h"

DEFINE_string(dataset, "karate", "Dataset to ML");
DEFINE_double(train_ratio, 0.5, "Train data ratio");
DEFINE_uint32(m, 10, "dimension of embedding");
DEFINE_uint32(T_in, 5, "T_in");
DEFINE_uint32(T_out, 10, "T_out");
DEFINE_double(tau, 1, "tau");
DEFINE_double(lambda, 1.0, "lambda");
DEFINE_double(mu, 0.01, "mu");
DEFINE_double(rho, 0.01, "rho");
DEFINE_bool(check_loss, false, "check loss every W learn");
DEFINE_uint64(seed, time(nullptr), "seed of random number");
DEFINE_string(model, "raw", "model name");
DEFINE_bool(check_W, false, "Output created W");

namespace {
  double h_dis(std::vector<size_t> const &answer,
               std::vector<size_t> const &predict){
    assert(answer.size() == predict.size());
    double result = 0;
    for(size_t i = 0; i < answer.size(); ++i){
      if(answer[i] != predict[i]){
        result += 1;
      }
    }

    return result / answer.size();
  }

  double h_dis(std::unordered_map<size_t, size_t> &answer,
               std::unordered_map<size_t, size_t> &predict){
    assert(answer.size() == predict.size());
    double result = 0;
    for(auto const &it: answer){
      if(predict[it.first] != it.second  /*= answer[it.first] */){
        result += 1;
      }
    }

    return result / answer.size();
  }
}

#ifdef NDEBUG
#error "If this shown, recreate assert function."
#endif

int main(int argc, char* argv[]){

  gflags::SetUsageMessage("DNE experiment");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)
    FLAGS_log_dir = R"(C:\Windows\Temp\)";
#endif
  google::InitGoogleLogging(argv[0]);

  std::cout << "OpenMP threads: " <<  Eigen::nbThreads() << std::endl;

  DatasetRepo::Dataset dataset =
    FLAGS_dataset == "karate"
    ? DatasetRepo::Karate
    : FLAGS_dataset == "youtube"
      ? DatasetRepo::YouTube
      : FLAGS_dataset == "flickr"
        ? DatasetRepo::Flickr
        : FLAGS_dataset == "wiki"
          ? DatasetRepo::Wiki
          : DatasetRepo::BlogCatalog;

  srand(FLAGS_seed);

  report("dataset: " + FLAGS_dataset);
  report("train_ratio: " + std::to_string(FLAGS_train_ratio));
  report("m: "+ std::to_string(FLAGS_m));
  report("T_in: " + std::to_string(FLAGS_T_in));
  report("T_out: " + std::to_string(FLAGS_T_out));
  report("tau: " + std::to_string(FLAGS_tau));
  report("lambda: " +  std::to_string(FLAGS_lambda));
  report("mu: " + std::to_string(FLAGS_mu));
  report("rho: " + std::to_string(FLAGS_rho));
  report("seed: " + std::to_string(FLAGS_seed));
  report("model: " + FLAGS_model);

  Params params(
    FLAGS_m, FLAGS_T_in, FLAGS_T_out,
    FLAGS_tau, FLAGS_lambda, FLAGS_mu,
    FLAGS_rho, FLAGS_seed, FLAGS_check_loss);

  Eigen::SparseMatrix<double, 0, std::ptrdiff_t> S;
  Eigen::SparseMatrix<double, 0, std::ptrdiff_t> L;
  std::unordered_map<size_t, size_t> T;
  std::vector<size_t> answer;
  size_t C;

  DatasetRepo::load(dataset, FLAGS_train_ratio, S, L, T, answer, C);
  assert(L.rows() == L.cols());

  Eigen::MatrixXd W,B;

  report("actual train ratio: " + std::to_string((double) T.size() / (double) answer.size()));

  if(FLAGS_model == "real"){
    RealDNE dne(S,T,C, params);
    dne.fit(W,B);
  }else if(FLAGS_model == "original"){
    OriginalDNE dne(S,T,C,params);
    dne.fit(W,B);
  }else if(FLAGS_model == "raw_svd"){
    RawSVD_DNE dne(S, T, C, params);
    dne.fit(W,B);
  }else if(FLAGS_model == "raw_semi"){
    RawSemiDNE dne(S, L, T, C, params);
    dne.fit(W, B);
  }else{
    assert(FLAGS_model == "raw");
    RawDNE dne(S,T,C, params);
    dne.fit(W,B);
  }

  assert((W.array() != NAN && W.array() != INFINITY).any());
  assert((B.array() != NAN && B.array() != INFINITY).any());

  auto N = S.rows();

  std::vector<size_t> predicted{};
  predicted.reserve(N);

  for(size_t n = 0; n < N; ++n){
    Eigen::Index max_index;
//    Eigen::MatrixXd pred;
//    binary_mult(W.transpose(), B.col(n), pred);
//    static_cast<Eigen::VectorXd>(pred).maxCoeff(&max_index);
    (W.transpose() * B.col(n)).maxCoeff(&max_index);
    predicted.push_back(max_index);
  }

  // ここで、trainとtestで別々に結果を見たい
  // そのためには、trainのindexとtestのindexの集合を別々に知る
  // train用のh-disとtest用のh-disをみたいね
  std::vector<size_t> train_index;
  std::vector<size_t> test_index;
  for(size_t n = 0; n < N; ++n){
    if(T.count(n) > 0){
      train_index.push_back(n);
    }else{
      test_index.push_back(n);
    }
  }

  // train/test用のanswer配列の生成
  std::unordered_map<size_t, size_t> answer_for_train;
  std::unordered_map<size_t, size_t> answer_for_test;
  for(auto train: train_index){
    answer_for_train[train] = answer[train];
  }
  for(auto test_i: test_index){
    answer_for_test[test_i] = answer[test_i];
  }

  std::unordered_map<size_t, size_t> predicted_train;
  for(auto train_i: train_index){
    Eigen::Index max_index;
    (W.transpose() * B.col(train_i)).maxCoeff(&max_index);
    predicted_train[train_i] = max_index;
  }
  std::unordered_map<size_t, size_t> predicted_test;
  for(auto test_i: test_index){
    Eigen::Index max_index;
    (W.transpose() * B.col(test_i)).maxCoeff(&max_index);
    predicted_test[test_i] = max_index;
  }

  double all_h_dis = h_dis(answer, predicted);
  double train_h_dis = h_dis(answer_for_train, predicted_train);
  double test_h_dis = h_dis(answer_for_test, predicted_test);
  double val = ( train_h_dis * T.size() + (test_h_dis * (N - T.size())) ) / N;
  assert(all_h_dis == val && "Train/Test divide algorithm is wrong!!");

  report("H-dis: " + std::to_string(all_h_dis));
  report("Train H-dis: " + std::to_string(train_h_dis));
  report("Test H-dis: " + std::to_string(test_h_dis));


  if(FLAGS_check_W){
    std::cout << B << std::endl;
  }

  Eigen::MatrixXd B_Bt;
  binary_mult_self(B, B_Bt);

  // || B * B^T ||^2
  std::cout << "constraint1: " << F_norm_pow2(B_Bt) << std::endl;
  // || B * 1 ||^2 //ここ計算おかしい！！！！！！！！！！
  std::cout << "constraint2: " << F_norm_pow2(B * Eigen::VectorXd::Ones(N)) << std::endl;
}