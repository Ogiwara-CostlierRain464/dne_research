#include "dne.h"
#include "loader.h"

namespace {
  template <typename T>
  double h_dis(std::vector<T> const &a,
               std::vector<T> const &b){
    assert(a.size() == b.size());
    double result = 0;
    for(size_t i = 0; i < a.size(); ++i){
      if(a[i] != b[i]){
        ++result;
      }
    }
    return result / a.size();
  }

  void karate(){
    std::vector<std::vector<size_t>> T = {
      {0}, {0}, {0}, {0}, {0},
      {0}, {0}, {0}, {0}, {1},
      {0}, {0}, {0}, {0}, {1}
    };
    UGraph g = from_file("../karate.adjlist");
    auto N = num_vertices(g);
    auto L = 15;
    auto C = 2;
    auto M = 50;
    Eigen::SparseMatrix<double> A(N, N);

    size_t from, to;
    for(auto i = vertices(g); i.first != i.second; ++i.first){
      from = *i.first;
      for(
        auto j=inv_adjacent_vertices(*i.first, g);
        j.first != j.second;
        ++j.first
        ){
        to = *j.first;
        A.insert(from, to) = 1.0;
      }
    }

    // use row wise op
    for(size_t i = 0; i < N; ++i){
      A.row(i) /= A.row(i).sum();
    }
    DNE dne(A, T, N, M, C, L, 5);
    Eigen::MatrixXd W, B;
    dne.fit(W, B);

    std::vector<std::vector<size_t>> correct = {
      {0}, {0}, {0}, {0}, {0},
      {0}, {0}, {0}, {0}, {1},
      {0}, {0}, {0}, {0}, {1},
      {1}, {0}, {0}, {1}, {0},
      {1}, {0}, {1}, {1}, {1},
      {1}, {1}, {1}, {1}, {1},
      {1}, {1}, {1}, {1}
    };
    std::vector<std::vector<size_t>> answer{};
    for(size_t n = 0; n < N; ++n){
      Eigen::Index max_index;
      (W.transpose() * B.col(n)).maxCoeff(&max_index);
      std::vector<size_t> v = {static_cast<size_t>(max_index)};
      answer.push_back(v);
    }

    std::cout << "H-dis: " << h_dis(answer, correct) << std::endl;
  }
}

int main(){
  karate();
}