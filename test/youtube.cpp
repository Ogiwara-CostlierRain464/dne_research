#include <gtest/gtest.h>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <nlohmann/json.hpp>

using namespace std;
using json = nlohmann::json;

class YouTube: public ::testing::Test{};

const string TARGET = "../dataset/youtube-links.txt";

TEST(YouTube, read_line){
  ifstream infile(TARGET);
  assert(infile);
  string line;
  int v1;
  int v2;
  typedef pair<int, int> Edge;
  vector<Edge> edges;

  while (getline(infile, line)){
    istringstream iss(line);
    if(iss >> v1 >> v2){
      edges.emplace_back(v1, v2);
    }
  }

  cout << edges.size() << endl;
}
