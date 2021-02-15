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

inline __attribute__((__always_inline__)) uint64_t h_dis(const __m256i a, const __m256i b){
  __m256i xor_ = _mm256_xor_epi64(a, b);
  return popcnt(&xor_, 256/8);
}
void p(){

  __m256i r0a = _mm256_set_epi64x(0b11111, 0b11111, 0b11111, 0b110);


  // まずEigen行列を256配列に
  // 一要素ごとに

  size_t R = 10000;
  size_t C = 256 * 30;
  Eigen::MatrixXd B(R,C);
  B.setRandom();
  B = B.unaryExpr([](float x){ return 2. * (x>=0) - 1.; });


  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();


  auto B_arr = new __m256i[(C/256) * R];


  for(size_t i = 0; i < R; ++i){
    for(size_t j = 0; j < C; j+=256){
      // 64個の数字ごとに格納
      // column orderであることに留意
      uint64_t part1 = 0;
      uint64_t part2 = 0;
      uint64_t part3 = 0;
      uint64_t part4 = 0;
      for(size_t k = 0; k < 64; ++k){
        unsigned int sign = (*(B.data() + i + (k+j) * R) > 0);
        part1 = (part1 << 1) | sign;
      }
      for(size_t k = 64; k < 128; ++k){
        unsigned int sign = (*(B.data() + i + (k+j) * R) > 0);
        part2 = (part2 << 1) | sign;
      }
      for(size_t k = 128; k < 192; ++k){
        unsigned int sign = (*(B.data() + i + (k+j) * R) > 0);
        part3 = (part3 << 1) | sign;
      }
      for(size_t k = 192; k < 256; ++k){
        unsigned int sign = (*(B.data() + i + (k+j) * R) > 0);
        part4 = (part4 << 1) | sign;
      }
      B_arr[i*(C/256) + (j/256)] = _mm256_set_epi64x(part1, part2, part3, part4);

//      if(i == 1 and j == 256){
//        std::bitset<64> x(part1);
//        std::cout << x << std::endl;
//        std::cout << part1 << std::endl;
//      }
//      B_arr[i*(C/256) + (j/256)] = _mm256_set_epi64x(part4, part3, part2, part1);

    }
  }

//  Log<uint64_t>(B_arr[1 * 2 + 1]);

  Eigen::MatrixXd ans(R,R);

  for(size_t i = 0; i < R; ++i){
    for(size_t j = 0; j < R; ++j){
      // inner product

      uint64_t h_dis_ij = 0;
      for(size_t part = 0; part < ( C / 256 );  ++part ){
        h_dis_ij += h_dis(B_arr[i * (C/256) + part], B_arr[ j * (C/256) + part ]);
      }

      int C_ = C;
      int h_dis_ij_ = h_dis_ij;

      ans.coeffRef(i,j) = C_ - 2 * h_dis_ij_;
    }
  }

  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

  std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[µs]" << std::endl;


  std::chrono::steady_clock::time_point begin2 = std::chrono::steady_clock::now();
  Eigen::MatrixXd ans1 = B * B.transpose();
  std::chrono::steady_clock::time_point end2 = std::chrono::steady_clock::now();

  std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::microseconds>(end2 - begin2).count() << "[µs]" << std::endl;

  assert(ans.sum() == ans1.sum());

//  __m256i r0a = _mm256_set_epi64x(0b11111, 0b11111, 0b11111, 0b110);
//  __m256i r0b = _mm256_set_epi64x(0, 0, 0, 0b000);
//  __m256i r1a = _mm256_set_epi64x(0b01011, 0b0111, 0b11111, 0b110);
//  __m256i r1b = _mm256_set_epi64x(0b00101010, 0, 0b11, 0b000);
//
//  __m256i A[] = { r0a, r0b, r1a, r1b };
//  uint64_t r = 512;
//
//  auto B_10 = r - 2 * (  h_dis(A[1*2+0], A[0+0]) + h_dis(A[1*2+1], A[0+1]) );
//  std::cout << B_10 ;

  exit(0);
}

int main(int argc, char* argv[]){
  p();

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