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
#include "dataset_repo.h"
#include "model/raw_dne.h"

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
}

int main(int argc, char* argv[]){
  gflags::SetUsageMessage("DNE experiment");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)
    FLAGS_log_dir = R"(C:\Windows\Temp\DNE)";
#endif
  google::InitGoogleLogging(argv[0]);

  std::cout << "OpenMP threads: " <<  Eigen::nbThreads() << std::endl;

  DatasetRepo::Dataset dataset = FLAGS_dataset == "karate"
    ? DatasetRepo::Karate
    : FLAGS_dataset == "youtube"
      ? DatasetRepo::YouTube
      : FLAGS_dataset == "flickr"
        ? DatasetRepo::Flickr
        : FLAGS_dataset == "wiki"
          ? DatasetRepo::Wiki
          : DatasetRepo::BlogCatalog;


  LOG(INFO) << "dataset: " << FLAGS_dataset;
  LOG(INFO) << "train_ratio: " << FLAGS_train_ratio;
  LOG(INFO) << "m: " << FLAGS_m;
  LOG(INFO) << "T_in: " << FLAGS_T_in;
  LOG(INFO) << "T_out: " << FLAGS_T_out;
  LOG(INFO) << "tau: " << FLAGS_tau;
  LOG(INFO) << "lambda: " << FLAGS_lambda;
  LOG(INFO) << "mu: " << FLAGS_mu;
  LOG(INFO) << "rho: " << FLAGS_rho;

  Eigen::SparseMatrix<double, 0, std::ptrdiff_t> S;
  std::unordered_map<size_t, size_t> T;
  std::vector<size_t> answer;
  size_t C;

  DatasetRepo::load(dataset, S, T, answer, C);

  RawDNE dne(S,T,C);
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

  std::cout << "H-dis: " << h_dis(answer, predicted) << std::endl;

}