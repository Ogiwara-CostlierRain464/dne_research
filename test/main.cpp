#include <gtest/gtest.h>
#include <atomic>
#include <bitset>
#include <random>
#include <fstream>
#include <Eigen/Dense>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/serialization/map.hpp>


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

    for(size_t j = 0; j < 0; ++j){
        assert(false);
    }
}

TEST_F(Unit, Heap){
    std::vector<int> a = {1,2,3,4,5};
    a.erase(std::remove(a.begin(), a.end(), 3), a.end());
    EXPECT_EQ(a[2], 4);

    std::unordered_map<size_t, std::vector<size_t>> b;
    std::vector<size_t> c = {1,2}, d = {};
    b.emplace(1, c);
    b.emplace(2, d);



    for(auto it = b.begin(); it != b.end();){
        if(it->second.empty()){
            it = b.erase(it);
        }else{
            ++it;
        }
    }
    EXPECT_EQ(b.size(), 1);
}

TEST_F(Unit, sign){
  Matrix3d a;
  a << 0, 1, 0,
      -1, 3, 0,
       3, 4, 9;

  cout << ceil(0 * 0.5) << endl;
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

TEST_F(Unit, save_map){

    std::map<int64_t, int64_t> foo{};
    foo.emplace(1,1);
    foo.emplace(3,2);

    std::ofstream ofs("map.txt");
    boost::archive::text_oarchive oa(ofs);
    oa << foo;
    ofs.close();

    std::map<int64_t, int64_t> foo2;
    std::ifstream  ifs("map.txt");
    boost::archive::text_iarchive ia(ifs);

    ia >> foo2;

    EXPECT_EQ(foo2[3], 2);


}

TEST_F(Unit, sa){
    std::unordered_map<size_t, std::vector<size_t>> a;
    std::vector<size_t> vec = {3,4};
    a.emplace(1, vec);

    EXPECT_EQ(a[2].empty(), true);
}

int main(int argc, char **argv){
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}