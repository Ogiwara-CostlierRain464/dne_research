#include "loader.h"
#include <Eigen/Sparse>
#include <Eigen/SparseCore>



using namespace std;
using namespace boost;
using namespace Eigen;

size_t N;
size_t M;
size_t C;
size_t L;
double TAU = 5;
double LAMBDA = 0.5;
double MU = 0.01;
double RHO = 0.01;

MatrixXd CF(MatrixXd const &x, MatrixXd const &y){
  return (x.array() == 0).select(y, x);
}

MatrixXd sgn(MatrixXd const &x){
  MatrixXd tmp = x.array().sign().matrix();
  return (tmp.array() == 0).select(-1, tmp);
}

MatrixXd WO(MatrixXd const &W, vector<size_t> const &T){
  assert(W.rows() == M and W.cols() == C);
  assert(T.size() == L);

  VectorXd sum_mc = W.rowwise().sum();
  MatrixXd wo = MatrixXd::Zero(M,N);
  for(size_t i = 0; i < L ; ++i){
    auto ci = T[i];
    wo.col(i) = sum_mc - (C * W.col(ci));
  }
  return wo;
}

MatrixXd eq11(MatrixXd const &B,
              MatrixXd const &S,
              MatrixXd const &W,
              vector<size_t> const &T){
  assert(B.rows() == M and B.cols() == N);
  assert(S.rows() == N and S.cols() == N);
  assert(W.rows() == M and W.cols() == C);
  MatrixXd dLB =- B * S
    + LAMBDA * WO(W, T)
    + MU * (B * B.transpose() * B)
    + RHO * (B * VectorXd::Ones(N) * RowVectorXd::Ones(N));

  return sgn(CF(TAU * B - dLB, B));
}

MatrixXd eq13(MatrixXd const &B, vector<size_t> const &T){
  MatrixXd W = MatrixXd::Zero(M, C);
  VectorXd b_sum = VectorXd::Zero(M);
  for(size_t i = 0; i < C; ++i){
    b_sum += B.col(i);
  }

  for(size_t c = 0; c < C; ++c){
    VectorXd sum_1 = VectorXd::Zero(M);
    for(size_t i = 0; i < L; ++i){
      if(T[i] == c)
        sum_1 += B.col(i);
    }

    VectorXd w_c = sgn(C * sum_1 - b_sum);
    W.col(c) = w_c;
  }
  return W;
}

double loss(MatrixXd const &B,
            MatrixXd const &S,
            MatrixXd const &W,
            vector<size_t> const &T){
  auto WO_ = WO(W, T);

  return - 0.5 * (B * S * B.transpose()).trace()
         + LAMBDA * (WO_.transpose() * B).trace()
         + MU * 0.25 * (B * B.transpose()).trace()
         + RHO * 0.5 * (B * VectorXd::Zero(N)).trace();
}

std::tuple<MatrixXd, MatrixXd, MatrixXd, MatrixXd>
discrete_network_embedding(MatrixXd const &A,
                           std::vector<size_t> const &T){
  MatrixXd S = (A + A * A) / 2;
  MatrixXd W = MatrixXd::Random(M, C);
  MatrixXd B = MatrixXd::Random(M, N);

  MatrixXd beforeW = W;
  MatrixXd beforeB = B;

  for(size_t _ = 1; _ <= 20; ++_){
    for(size_t i = 1; i <= 9; ++i){
      B = eq11(B, S, W, T);
      cout << "updating B " << loss(B, S, W, T) << endl;
    }
    W = eq13(B, T);
    cout << "updating W " << loss(B, S, W, T) << endl;
  }

  return make_tuple(beforeB, beforeW, B, W);
}

double h_dis(vector<size_t> const &a, vector<size_t> const &b){
  assert(a.size() == b.size());
  double result = 0;
  for(size_t i = 0; i < a.size(); ++i){
    if(a[i] != b[i]){
      ++result;
    }
  }
  return result / a.size();
}

int main(){
  vector<size_t> T = {
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 1,
    0, 0, 0, 0, 1
  };

  UGraph g = from_file("../karate.adjlist");
  N = num_vertices(g);
  L = 15;
  C = 2;
  M = 100;

  Eigen::SparseMatrix<double> SparseA(N, N);

  size_t from, to;
  for(auto i = vertices(g); i.first != i.second; ++i.first){
    from = *i.first;
    for(
      auto j=inv_adjacent_vertices(*i.first, g);
      j.first != j.second;
      ++j.first
    ){
      to = *j.first;
      SparseA.insert(from, to) = 1.0;
    }
  }


  MatrixXd A = MatrixXd(SparseA);

  // use row wise op
  for(size_t i = 0; i < N; ++i){
    A.row(i) /= A.row(i).sum();
  }

  auto tuple = discrete_network_embedding(A, T);
  auto beforeB = get<0>(tuple);
  auto beforeW = get<1>(tuple);
  auto B = get<2>(tuple);
  auto W = get<3>(tuple);
  vector<size_t> correct = {
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 1,
    0, 0, 0, 0, 1,
    1, 0, 0, 1, 0,
    1, 0, 1, 1, 1,
    1, 1, 1, 1, 1,
    1, 1, 1, 1
  };
  vector<size_t> answer{};
  for(size_t n = 0; n < N; ++n){
    Index max_index;
    (W.transpose() * B.col(n)).maxCoeff(&max_index);
    answer.push_back(max_index);
  }

  MatrixXd S = (A + (A * A)) / 2;

  double loss_before = loss(beforeB, S, beforeW, T);
  double loss_after = loss(B, S, W, T);

  cout << "LOSS gain: " << (loss_after - loss_before) << endl;
  cout << "H-dis: " << h_dis(answer, correct) << endl;

}