#include <gtest/gtest.h>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>

using namespace std;
using namespace boost::property_tree;

class JSON: public ::testing::Test{};

TEST(JSON, load){
  ptree pt;

  read_json("/Users/ogiwara/Downloads/dblp.v12.json", pt);
//  EXPECT_EQ(pt.get_optional<int>("Data.value"), 3);
}

