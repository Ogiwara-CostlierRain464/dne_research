#include <gtest/gtest.h>
#include <atomic>
#include <bitset>
#include <random>

using namespace std;

class Unit: public ::testing::Test{};

TEST_F(Unit, init){
  ASSERT_EQ(1+1, 2);
}


int main(int argc, char **argv){
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}