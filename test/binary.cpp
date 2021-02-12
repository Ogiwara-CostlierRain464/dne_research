#include <gtest/gtest.h>
#include "../src/BMP.h"
#include <Eigen/Core>
using Eigen::MatrixXf;
using namespace std;


class Binary: public ::testing::Test{};


float float_sign(float x){
  // return (x>=0);
  return 2. * (x>=0) - 1.;
}

TEST(Binary, test){
  std::cout << "OpenMP threads: " <<  Eigen::nbThreads() << std::endl;

  int N = 4096 * 4;

  MatrixXf A(N,N);
  // A.setZero();
  A.setRandom();
  A = A.unaryExpr(ptr_fun(float_sign));
  // cout <<endl<<"A max = " <<A.maxCoeff();
  // cout <<endl<<"A min = " <<A.minCoeff();
  // cout <<endl<<"A sum = " <<A.sum();

  MatrixXf B(N,N);
  // B.setZero();
  B.setRandom();
  B = B.unaryExpr(ptr_fun(float_sign));
  // cout <<endl<<"B max = " <<B.maxCoeff();
  // cout <<endl<<"B min = " <<B.minCoeff();

  // cout <<endl<<"A B diff = " <<(A-B).sum();

  std::cout << B.topLeftCorner(10, 10) << std::endl;

  MatrixXf C1(N,N);
  MatrixXf C2(N,N);

  double elapsed_time = omp_get_wtime();
  C1 = BMP(A,B);
  // C1 = A*B;
  elapsed_time = omp_get_wtime()-elapsed_time;
  cout <<endl<<"BPM elapsed_time = " << elapsed_time<<"s";

  elapsed_time = omp_get_wtime();
  C2 = A*B;
  // C2 = BMP(A,B);
  elapsed_time = omp_get_wtime()-elapsed_time;
  cout <<endl<<"Eigen SGEMM elapsed_time = " << elapsed_time<<"s";

  cout<<endl<<"C1 sum = " << C1.sum();
  cout<<endl<<"C2 sum = " << C2.sum();
  cout<<endl<<"Mean difference = " << (C1-C2).mean()<<endl<<endl;
}

TEST(Binary, OpenMP_tutorial){
  int tmp = 0;
#pragma omp for private(tmp)
  for(int j = 0; j < 1000; ++j){
    tmp += j;
  }
  printf("%d\n", tmp);
}