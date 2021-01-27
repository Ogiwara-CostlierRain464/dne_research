#include <gtest/gtest.h>
#include <atomic>
#include <bitset>
#include <random>
#include <Eigen/Dense>


using namespace std;
using namespace Eigen;

class Unit: public ::testing::Test{};

TEST_F(Unit, init){
  ASSERT_EQ(1+1, 2);
}

TEST_F(Unit, select){
  Matrix3d a;
  a << 0, 1, 0,
      -1, 3, 0,
       3, 4, 9;

  Matrix3d b;
  b << 9, 8, 6,
       4, 7, 1,
       0, -3, 1;

  cout << (a.array() == 0).select(
    /*then*/ b,
    /*else*/ a) << endl;
}

TEST_F(Unit, sign){
  Matrix3d a;
  a << 0, 1, 0,
      -1, 3, 0,
       3, 4, 9;

  cout << a.array().sign() << endl;
}

TEST_F(Unit, shape){
  Matrix3d a;
  a << 0, 1, 0,
      -1, 3, 0,
       3, 4, 9;

  EXPECT_EQ(a.rows(), 3);
  EXPECT_EQ(a.cols(), 3);

}

TEST_F(Unit, colwise_sum){
  Matrix3d a;
  a << 0, 1, 0,
      -1, 3, 0,
       3, 4, 9;
//  cout << a.rowwise().sum() << endl;
  cout << a.array().mean() << endl;
}

TEST_F(Unit, max_index){
  Vector3d a;
  a << 1, 3, 2;
  Index i;
  a.maxCoeff(&i);
  EXPECT_EQ(i, 1);
}

TEST_F(Unit, random){
  auto m = MatrixXd::Random(10000, 10000);
  auto j = m * m;
  cout << j.col(0) << endl;
}

int main(int argc, char **argv){
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}