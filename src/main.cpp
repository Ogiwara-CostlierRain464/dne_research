#include "dne.h"
#include "real_dne.h"
#include "propose_dne.h"
#include "propose_dne2.h"
#include "loader.h"
#include "experiment_dne.h"
#include "original_dne.h"
#include "dataset_repo.h"
#include <boost/graph/graphviz.hpp>
#include <gflags/gflags.h>
#include <glog/logging.h>


DEFINE_string(message, "Hello world!", "Message to print");
DEFINE_bool(c, true, "Test message!");

namespace {
  template <typename T>
  double h_dis(std::vector<T> const &a,
               std::vector<T> const &b){
    assert(a.size() == b.size());
    double result = 0;
    for(size_t i = 0; i < a.size(); ++i){
      if(a[i] != b[i]){
        result += 1;
      }
    }
    return result / a.size();
  }

  void karate(){
    UGraph g;
    std::unordered_map<size_t, size_t> T;
    std::vector<size_t> answer;

    auto N = 34;
    auto C = 2;
    from_txt("../dataset/karate.txt", N, C, 0.5, g, T, answer);
    auto L = T.size();
    auto M = 10;
    DNE::Sp A(N, N);

    typedef boost::property_map<UGraph, boost::vertex_index_t>::type IndexMap;
    IndexMap index = get(boost::vertex_index, g);
    typedef boost::graph_traits<UGraph> GraphTraits;
    typename GraphTraits::edge_iterator ei, ei_end;
    for(tie(ei, ei_end) = edges(g); ei != ei_end; ++ei){
      auto sur = index[boost::source(*ei, g)];
      auto tar = index[boost::target(*ei, g)];

      A.insert(sur, tar) = 1.0;
      A.insert(tar, sur) = 1.0;
    }

//    std::cout << A << std::endl;

    assert(A.isApprox(A.transpose()));


    // use row wise op
    #pragma omp parallel for
    for(size_t i = 0; i < N; ++i){
      A.row(i) /= A.row(i).sum();
    }

    DNE dne(A, T, N, M, C, L, 5);
    Eigen::MatrixXd W, B;
    dne.fit(W, B);

    std::vector<size_t> predicted{};
    predicted.reserve(N);
    for(size_t n = 0; n < N; ++n){
      Eigen::Index max_index;
      (W.transpose() * B.col(n)).maxCoeff(&max_index);
      predicted.push_back(max_index);
    }

    std::cout << "H-dis: " << h_dis(answer, predicted) << std::endl;
  }

  void karate2(){
      UGraph g;
      std::unordered_map<size_t, size_t> T;
      std::vector<size_t> answer;


  }

  void youtube(){
    UGraph g;
    std::unordered_map<size_t, size_t> T;
    std::vector<size_t> answer;

    auto N = 1138499;
    auto C = 47;
    from_txt("../dataset/youtube.txt", N, C, 0.6, g, T, answer);
    auto L = T.size();
    auto M = 500;
    DNE::Sp A(N, N);

    typedef boost::property_map<UGraph, boost::vertex_index_t>::type IndexMap;
    IndexMap index = get(boost::vertex_index, g);
    typedef boost::graph_traits<UGraph> GraphTraits;
    typename GraphTraits::edge_iterator ei, ei_end;
    for(tie(ei, ei_end) = edges(g); ei != ei_end; ++ei){
      auto sur = index[boost::source(*ei, g)];
      auto tar = index[boost::target(*ei, g)];

      A.insert(sur, tar) = 1.0;
    }

//    A.makeCompressed();
//    assert(A.isApprox(A.transpose()));

    #pragma omp parallel for
    for(size_t i = 0; i < N; ++i){
      A.row(i) /= A.row(i).sum();
    }

    DNE dne(A, T, N, M, C, L, 5);
    Eigen::MatrixXd W, B;
    dne.fit(W, B);

    std::vector<size_t> predicted{};
    predicted.reserve(N);
    for(size_t n = 0; n < N; ++n){
      Eigen::Index max_index;
      (W.transpose() * B.col(n)).maxCoeff(&max_index);
      predicted.push_back(max_index);
    }

    std::cout << "H-dis: " << h_dis(answer, predicted) << std::endl;
  }

  void flicker(){
        UGraph g;
        std::unordered_map<size_t, size_t> T;
        std::vector<size_t> answer;

        auto N = 80513;
        auto C = 195;
        from_txt("../dataset/flickr.txt", N, C, 0.6, g, T, answer);
        auto L = T.size();
        auto M = 500;
        DNE::Sp A(N, N);

        typedef boost::property_map<UGraph, boost::vertex_index_t>::type IndexMap;
        IndexMap index = get(boost::vertex_index, g);
        typedef boost::graph_traits<UGraph> GraphTraits;
        typename GraphTraits::edge_iterator ei, ei_end;
        for(tie(ei, ei_end) = edges(g); ei != ei_end; ++ei){
            auto sur = index[boost::source(*ei, g)];
            auto tar = index[boost::target(*ei, g)];

            A.insert(sur, tar) = 1.0;
        }

//    A.makeCompressed();
        assert(A.isApprox(A.transpose()));

        #pragma omp parallel for
        for(size_t i = 0; i < N; ++i){
            A.row(i) /= A.row(i).sum();
        }

        DNE dne(A, T, N, M, C, L, 5);
        Eigen::MatrixXd W, B;
        dne.fit(W, B);

        std::vector<size_t> predicted{};
        predicted.reserve(N);
        for(size_t n = 0; n < N; ++n){
            Eigen::Index max_index;
            (W.transpose() * B.col(n)).maxCoeff(&max_index);
            predicted.push_back(max_index);
        }

        std::cout << "H-dis: " << h_dis(answer, predicted) << std::endl;
    }

  void catalog(){
    UGraph g;
    std::unordered_map<size_t, size_t> T;
    std::vector<size_t> answer;

    // Edge Number: 667932
    // x64.77 density. (or 32?)
    auto N = 10312;
    auto C = 39;
    from_txt("../dataset/blogcatalog.txt", N, C, 0.5, g, T, answer);
    auto L = T.size();
    auto M = 128;
    Eigen::SparseMatrix<double> A(N, N);

    typedef boost::property_map<UGraph, boost::vertex_index_t>::type IndexMap;
    IndexMap index = get(boost::vertex_index, g);
    typedef boost::graph_traits<UGraph> GraphTraits;
    typename GraphTraits::edge_iterator ei, ei_end;
    for(tie(ei, ei_end) = edges(g); ei != ei_end; ++ei){
      auto sur = index[boost::source(*ei, g)];
      auto tar = index[boost::target(*ei, g)];

      A.insert(sur, tar) = 1.0;
    }

//    assert(A.isApprox(A.transpose()));


    // use row wise op
    for(size_t i = 0; i < N; ++i){
      A.row(i) /= A.row(i).sum();
    }

    DNE dne(A, T, N, M, C, L, 5);
    Eigen::MatrixXd W, B;
    dne.fit(W, B);

    std::vector<size_t> predicted{};
    predicted.reserve(N);
    for(size_t n = 0; n < N; ++n){
      Eigen::Index max_index;
      (W.transpose() * B.col(n)).maxCoeff(&max_index);
      predicted.push_back(max_index);
    }

    std::cout << "H-dis: " << h_dis(answer, predicted) << std::endl;
  }

  void wiki(){
    UGraph g;
    std::unordered_map<size_t, size_t> T;
    std::vector<size_t> answer;

    // Edge Number: 17981
    // C = 16
    auto N = 2405;
    auto C = 17;
    from_txt2("../dataset/Wiki_category.txt",
              "../dataset/Wiki_edgelist.txt",
              N, C, 0.5, g, T, answer);
    auto L = T.size();
    auto M = 128;
    DNE::Sp A(N, N);

    typedef boost::property_map<UGraph, boost::vertex_index_t>::type IndexMap;
    IndexMap index = get(boost::vertex_index, g);
    typedef boost::graph_traits<UGraph> GraphTraits;
    typename GraphTraits::edge_iterator ei, ei_end;
    for(tie(ei, ei_end) = edges(g); ei != ei_end; ++ei){
      auto sur = index[boost::source(*ei, g)];
      auto tar = index[boost::target(*ei, g)];

      A.coeffRef(sur, tar) = 1.0;
      A.coeffRef(tar, sur) = 1.0;
    }

    assert(A.isApprox(A.transpose()));


    // use row wise op
    for(size_t i = 0; i < N; ++i){
      A.row(i) /= A.row(i).sum();
    }

    DNE dne(A, T, N, M, C, L, 3);
//    RealDNE dne(A, T, N, M, C, L, 10);
    Eigen::MatrixXd W, B;
    dne.fit(W, B);

    std::vector<size_t> predicted{};
    predicted.reserve(N);
    for(size_t n = 0; n < N; ++n){
      Eigen::Index max_index;
      (W.transpose() * B.col(n)).maxCoeff(&max_index);
      predicted.push_back(max_index);
    }

    std::cout << "H-dis: " << h_dis(answer, predicted) << std::endl;
  }

  void propose(){
    UGraph g;
    std::unordered_map<size_t, size_t> T;
    std::vector<size_t> answer;

    auto N = 34;
    auto C = 2;
    from_txt("../dataset/karate.txt", N, C, 0.5, g, T, answer);
    auto L = T.size();
    auto M = 20;
    DNE::Sp A(N, N);

    typedef boost::property_map<UGraph, boost::vertex_index_t>::type IndexMap;
    IndexMap index = get(boost::vertex_index, g);
    typedef boost::graph_traits<UGraph> GraphTraits;
    typename GraphTraits::edge_iterator ei, ei_end;
    for(tie(ei, ei_end) = edges(g); ei != ei_end; ++ei){
      auto sur = index[boost::source(*ei, g)];
      auto tar = index[boost::target(*ei, g)];

      A.insert(sur, tar) = 1.0;
      A.insert(tar, sur) = 1.0;
    }

//    std::cout << A << std::endl;

    assert(A.isApprox(A.transpose()));


    // use row wise op
    #pragma omp parallel for
    for(size_t i = 0; i < N; ++i){
      A.row(i) /= A.row(i).sum();
    }

    ProposeDNE2 dne(A, T, N, M, C, L, 2);
    Eigen::MatrixXd W, B;
    dne.fit(W, B);

    std::vector<size_t> predicted{};
    predicted.reserve(N);
    for(size_t n = 0; n < N; ++n){
      Eigen::Index max_index;
      (W.transpose() * B.col(n)).maxCoeff(&max_index);
      predicted.push_back(max_index);
    }

    std::cout << "H-dis: " << h_dis(answer, predicted) << std::endl;
  }

  void propose2(){
    UGraph g;
    std::unordered_map<size_t, size_t> T;
    std::vector<size_t> answer;

    // Edge Number: 17981
    // C = 16
    auto N = 2405;
    auto C = 17;
    from_txt2("../dataset/Wiki_category.txt",
              "../dataset/Wiki_edgelist.txt",
              N, C, 0.5, g, T, answer);
    auto L = T.size();
    auto M = 128;
    DNE::Sp A(N, N);

    typedef boost::property_map<UGraph, boost::vertex_index_t>::type IndexMap;
    IndexMap index = get(boost::vertex_index, g);
    typedef boost::graph_traits<UGraph> GraphTraits;
    typename GraphTraits::edge_iterator ei, ei_end;
    for(tie(ei, ei_end) = edges(g); ei != ei_end; ++ei){
      auto sur = index[boost::source(*ei, g)];
      auto tar = index[boost::target(*ei, g)];

      A.coeffRef(sur, tar) = 1.0;
      A.coeffRef(tar, sur) = 1.0;
    }

//    std::cout << A << std::endl;

    assert(A.isApprox(A.transpose()));


    // use row wise op
#pragma omp parallel for
    for(size_t i = 0; i < N; ++i){
      A.row(i) /= A.row(i).sum();
    }

    ProposeDNE2 dne(A, T, N, M, C, L, 1);
    Eigen::MatrixXd W, B;
    dne.fit(W, B);

    std::vector<size_t> predicted{};
    predicted.reserve(N);
    for(size_t n = 0; n < N; ++n){
      Eigen::Index max_index;
      (W.transpose() * B.col(n)).maxCoeff(&max_index);
      predicted.push_back(max_index);
    }

    std::cout << "H-dis: " << h_dis(answer, predicted) << std::endl;
  }

  void experiment1(){
    UGraph g;
    std::unordered_map<size_t, size_t> T;
    std::vector<size_t> answer;

    auto N = 34;
    auto C = 2;
    from_txt("../dataset/karate.txt", N, C, 0.5, g, T, answer);
    auto L = T.size();
    auto M = 10;
    DNE::Sp A(N, N);

    typedef boost::property_map<UGraph, boost::vertex_index_t>::type IndexMap;
    IndexMap index = get(boost::vertex_index, g);
    typedef boost::graph_traits<UGraph> GraphTraits;
    typename GraphTraits::edge_iterator ei, ei_end;
    for(tie(ei, ei_end) = edges(g); ei != ei_end; ++ei){
      auto sur = index[boost::source(*ei, g)];
      auto tar = index[boost::target(*ei, g)];

      A.insert(sur, tar) = 1.0;
      A.insert(tar, sur) = 1.0;
    }

    assert(A.isApprox(A.transpose()));

    // use row wise op
#pragma omp parallel for
    for(size_t i = 0; i < N; ++i){
      A.row(i) /= A.row(i).sum();
    }

    OriginalDNE dne(A, T, N, M, C, L, 5);
    Eigen::MatrixXd W, B;
    dne.fit(W, B);

    std::vector<size_t> predicted{};
    predicted.reserve(N);
    for(size_t n = 0; n < N; ++n){
      Eigen::Index max_index;
      (W.transpose() * B.col(n)).maxCoeff(&max_index);
      predicted.push_back(max_index);
    }

    std::cout << "H-dis: " << h_dis(answer, predicted) << std::endl;
  }

  void experiment2(){
    UGraph g;
    std::unordered_map<size_t, size_t> T;
    std::vector<size_t> answer;

    // Edge Number: 17981
    // C = 16
    auto N = 2405;
    auto C = 17;
    from_txt2("../dataset/Wiki_category.txt",
              "../dataset/Wiki_edgelist.txt",
              N, C, 0.5, g, T, answer);
    auto L = T.size();
    auto M = 128;
    DNE::Sp A(N, N);

    typedef boost::property_map<UGraph, boost::vertex_index_t>::type IndexMap;
    IndexMap index = get(boost::vertex_index, g);
    typedef boost::graph_traits<UGraph> GraphTraits;
    typename GraphTraits::edge_iterator ei, ei_end;
    for(tie(ei, ei_end) = edges(g); ei != ei_end; ++ei){
      auto sur = index[boost::source(*ei, g)];
      auto tar = index[boost::target(*ei, g)];

      A.coeffRef(sur, tar) = 1.0;
      A.coeffRef(tar, sur) = 1.0;
    }
//    std::cout << A << std::endl;

    assert(A.isApprox(A.transpose()));


    // use row wise op
#pragma omp parallel for
    for(size_t i = 0; i < N; ++i){
      A.row(i) /= A.row(i).sum();
    }

    OriginalDNE dne(A, T, N, M, C, L, 1);
    Eigen::MatrixXd W, B;
    dne.fit(W, B);

    std::vector<size_t> predicted{};
    predicted.reserve(N);
    for(size_t n = 0; n < N; ++n){
      Eigen::Index max_index;
      (W.transpose() * B.col(n)).maxCoeff(&max_index);
      predicted.push_back(max_index);
    }

    std::cout << "H-dis: " << h_dis(answer, predicted) << std::endl;
  }
}

int main(int argc, char* argv[]){
    // 実験を円滑に行うために、パラメータを受け取れるようにすべきである。
    // それはともかく、元のDNE class自体がそうすべき？
    // もうgflagsを導入するか～、glogもね
    // glogではどうするか？とりあえず実験の取り方をもうちょっとは楽にしたい
    // パラメータはコマンドラインから指定？で結果はすべて出力
    // とはいえど、すぐに実装はかえたいよね
    // 平均とかどうとるのさ、プログラム中でfor回していくとかだろうねえ

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)
    FLAGS_log_dir = R"(C:\Windows\Temp\DNE)";
#endif

    google::InitGoogleLogging(argv[0]);




#ifdef DEBUG
  printf("This is DEBUG mode.\n");
#endif

  std::cout << FLAGS_message << std::endl;
  std::cout << Eigen::nbThreads() << std::endl;

//  catalog();
//  youtube();
//  karate();
//  wiki();
//  propose();
//  propose2();

//  experiment1();
//  experiment2();
//    flicker();
    Eigen::SparseMatrix<double, 0, std::ptrdiff_t> S;
    std::unordered_map<size_t, size_t> T;
    std::vector<size_t> answer;
    DatasetRepo::loadS(DatasetRepo::BlogCatalog, S, T, answer);
}