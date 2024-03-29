#include <gtest/gtest.h>
#include "../src/BMP.h"
#include "../src/binary.h"
#include <Eigen/Core>
#include <chrono>
using Eigen::MatrixXf;
using namespace std;
using namespace std::literals::chrono_literals;

class Binary: public ::testing::Test{};


float float_sign(float x){
  // return (x>=0);
  return 2. * (x>=0) - 1.;
}

TEST(Binary, test){
  std::cout << "OpenMP threads: " <<  Eigen::nbThreads() << std::endl;

  int N = 4096 ;

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
#pragma omp parallel for private(tmp)
  for(int j = 0; j < 1000; ++j){
    tmp += j;
  }
  printf("%d\n", tmp);
}

TEST(Binary, my){
  Eigen::MatrixXd B(128, 1'000'000);
  B.setRandom();
  B = B.unaryExpr([](float x){ return 2. * (x>=0) - 1.; });

  std::chrono::steady_clock::time_point start1 = std::chrono::steady_clock::now();
  Eigen::MatrixXd B_Bt;
//  binary_mult(B,B.transpose(), B_Bt);
  binary_mult_self(B, B_Bt);
  Eigen::MatrixXd actual = B_Bt * B;

  std::chrono::steady_clock::time_point end1 = std::chrono::steady_clock::now();
  std::cout << "AVX2 " << "Time difference = "
            << std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1).count() << "[micro-s]" << std::endl;

  std::chrono::steady_clock::time_point start3 = std::chrono::steady_clock::now();
  Eigen::MatrixXd B_Bt_;
//  binary_mult(B,B.transpose(), B_Bt);
  binary_mult512_self(B, B_Bt);
  Eigen::MatrixXd actual_ = B_Bt * B;

  std::chrono::steady_clock::time_point end3 = std::chrono::steady_clock::now();
  std::cout << "AVX512 " << "Time difference = "
            << std::chrono::duration_cast<std::chrono::microseconds>(end3 - start3).count() << "[micro-s]" << std::endl;


  std::chrono::steady_clock::time_point start2 = std::chrono::steady_clock::now();
  Eigen::MatrixXd expected = (B * B.transpose()) * B;
  std::chrono::steady_clock::time_point end2 = std::chrono::steady_clock::now();
  std::cout << "Intel SGEMM " << "Time difference = "
            << std::chrono::duration_cast<std::chrono::microseconds>(end2 - start2).count() << "[micro-s]" << std::endl;


  EXPECT_EQ(actual.sum(), expected.sum());
  EXPECT_EQ(actual_.sum(), expected.sum());
}


TEST(Binary, my2){
  Eigen::MatrixXd A(10,15);
  A.setRandom();
  A = A.unaryExpr([](float x){ return 2. * (x>=0) - 1.; });
  Eigen::MatrixXd B(15,12);
  B.setRandom();
  B = B.unaryExpr([](float x){ return 2. * (x>=0) - 1.; });

  Eigen::MatrixXd C;
  binary_mult(A, B , C);

  EXPECT_EQ(C, A * B);

}

TEST(Binary, my3){
  Eigen::MatrixXd B(100, 256000);
  B.setRandom();
  B = B.unaryExpr([](float x){ return 2. * (x>=0) - 1.; });

  std::chrono::steady_clock::time_point start2 = std::chrono::steady_clock::now();
  Eigen::MatrixXd expected = (B * B.transpose());
  std::chrono::steady_clock::time_point end2 = std::chrono::steady_clock::now();
  std::cout << "Intel SGEMM " << "Time difference = "
            << std::chrono::duration_cast<std::chrono::microseconds>(end2 - start2).count() << "[micro-s]" << std::endl;



  std::chrono::steady_clock::time_point start1 = std::chrono::steady_clock::now();
  Eigen::MatrixXd actual;
//  binary_mult(B,B.transpose(), B_Bt);
  binary_mult_self(B, actual);

  std::chrono::steady_clock::time_point end1 = std::chrono::steady_clock::now();
  std::cout << "AVX2 " << "Time difference = "
            << std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1).count() << "[micro-s]" << std::endl;

  std::chrono::steady_clock::time_point start3 = std::chrono::steady_clock::now();
  Eigen::MatrixXd actual_;
//  binary_mult(B,B.transpose(), B_Bt);
  binary_mult512_self(B, actual_);

  std::chrono::steady_clock::time_point end3 = std::chrono::steady_clock::now();
  std::cout << "AVX512 " << "Time difference = "
            << std::chrono::duration_cast<std::chrono::microseconds>(end3 - start3).count() << "[micro-s]" << std::endl;



  EXPECT_EQ(actual.sum(), expected.sum());
  EXPECT_EQ(actual_.sum(), expected.sum());
}