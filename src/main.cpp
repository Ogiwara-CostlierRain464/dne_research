#include "dne.h"
#include "loader.h"
#include <boost/graph/graphviz.hpp>


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

    std::unordered_map<size_t, size_t> T{
      {0, 0}, {1, 0}, {2, 0}, {3, 0}, {4, 0},
      {5, 0}, {6, 0}, {7, 0}, {8, 0}, {9, 1},
      {10, 0},{11,0}, {12, 0}, {13,0},{14,1},
      {32, 1}, {31, 1}, {30, 1}
    };

    UGraph g = from_file("../karate.adjlist");
    auto N = num_vertices(g);
    auto L = T.size();
    auto C = 2;
    auto M = 50;
    Eigen::SparseMatrix<double> A(N, N);

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
    for(size_t i = 0; i < N; ++i){
      A.row(i) /= A.row(i).sum();
    }
    DNE dne(A, T, N, M, C, L, 5);
    Eigen::MatrixXd W, B;
    dne.fit(W, B);

    std::vector<size_t> correct = {
      0, 0, 0, 0, 0,
      0, 0, 0, 0, 1,
      0, 0, 0, 0, 1,
      1, 1, 1, 1, 1,
      1, 0, 1, 1, 1,
      1, 1, 1, 1, 1,
      1, 1, 1, 1
    };
    std::vector<size_t> answer{};
    for(size_t n = 0; n < N; ++n){
      Eigen::Index max_index;
      (W.transpose() * B.col(n)).maxCoeff(&max_index);
      answer.push_back(max_index);
    }

    std::cout << "H-dis: " << h_dis(answer, correct) << std::endl;
  }

  void gen(){
    UGraph g;
    std::unordered_map<size_t, size_t> T;
    std::vector<size_t> answer;

    auto N = 34;
    auto C = 2;
    from_txt("../dataset/karate.txt", N, C, 0.4, g, T, answer);
    auto L = T.size();
    auto M = 50;
    Eigen::SparseMatrix<double> A(N, N);

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

    std::cout << A << std::endl;

    assert(A.isApprox(A.transpose()));


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


  void catalog(){
    UGraph g;
    std::unordered_map<size_t, size_t> T;
    std::vector<size_t> answer;

    auto N = 10312;
    auto C = 39;
    from_txt("../dataset/blogcatalog.txt", N, C, 0.3, g, T, answer);
    auto L = T.size();
    auto M = 50;
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

    assert(A.isApprox(A.transpose()));


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
}

int main(){
#ifndef NDEBUG
  printf("This is DEBUG mode.\n");
#endif
  catalog();
}