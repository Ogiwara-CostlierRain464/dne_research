#include <boost/graph/graphviz.hpp>
#include <boost/serialization/unordered_map.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/map.hpp>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <Eigen/Sparse>
#include <Eigen/SparseCore>
#include <Eigen/SVD>
#include <immintrin.h>
#include "../third_party/libpopcnt/libpopcnt.h"
#include "dataset_repo.h"
#include "model/raw_dne.h"
#include "model/original_dne.h"
#include "model/real_dne.h"

DEFINE_string(dataset, "karate", "Dataset to ML");
DEFINE_double(train_ratio, 0.5, "Train data ratio");
DEFINE_uint32(m, 10, "dimension of embedding");
DEFINE_uint32(T_in, 5, "T_in");
DEFINE_uint32(T_out, 10, "T_out");
DEFINE_double(tau, 0.01, "tau");
DEFINE_double(lambda, 1.0, "lambda");
DEFINE_double(mu, 0.01, "mu");
DEFINE_double(rho, 0.01, "rho");

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

  void report(std::string const &log){
    LOG(INFO) << log;
    std::cout << log << std::endl;
  }
}

template<class T> inline void Log(const __m256i & value)
{
  const size_t n = sizeof(__m256i) / sizeof(T);
  T buffer[n];
  _mm256_storeu_si256((__m256i*)buffer, value);
  for (int i = n - 1; i > -1 ; --i)
    std::cout << buffer[i] << " ";
  std::cout << std::endl;
}


int main(int argc, char* argv[]){

  gflags::SetUsageMessage("DNE experiment");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)
    FLAGS_log_dir = R"(C:\Windows\Temp\DNE)";
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


  report("dataset: " + FLAGS_dataset);
  report("train_ratio: " + std::to_string(FLAGS_train_ratio));
  report("m: "+ std::to_string(FLAGS_m));
  report("T_in: " + std::to_string(FLAGS_T_in));
  report("T_out: " + std::to_string(FLAGS_T_out));
  report("tau: " + std::to_string(FLAGS_tau));
  report("lambda: " +  std::to_string(FLAGS_lambda));
  report("mu: " + std::to_string(FLAGS_mu));
  report("rho: " + std::to_string(FLAGS_rho));

  Params params(
    FLAGS_m, FLAGS_T_in, FLAGS_T_out,
    FLAGS_tau, FLAGS_lambda, FLAGS_mu,
    FLAGS_rho, time(nullptr));

  Eigen::SparseMatrix<double, 0, std::ptrdiff_t> S;
  std::unordered_map<size_t, size_t> T;
  std::vector<size_t> answer;
  size_t C;

  DatasetRepo::load(dataset, FLAGS_train_ratio, S, T, answer, C);

  RealDNE dne(S,T,C, params);
  Eigen::MatrixXd W,B;
  dne.fit(W,B);

  auto N = S.rows();

  std::vector<size_t> predicted{};
  predicted.reserve(N);
  for(size_t n = 0; n < N; ++n){
    Eigen::Index max_index;
    (W.transpose() * B.col(n)).maxCoeff(&max_index);
    predicted.push_back(max_index);
  }


  // さて、交差確認やlossの減少の確認は？
  // そこらへんもゆくゆくは整備

  std::cout << "H-dis: " << h_dis(answer, predicted) << std::endl;

}