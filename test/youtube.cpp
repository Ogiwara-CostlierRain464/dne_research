#include <gtest/gtest.h>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <nlohmann/json.hpp>
#include <boost/graph/undirected_graph.hpp>

#include <list>

using namespace std;
using json = nlohmann::json;

class YouTube: public ::testing::Test{};

const string TARGET = "../dataset/youtube-links.txt";

TEST(YouTube, blogcatalog){
  ifstream infile("../dataset/blogcatalog.txt");
  // file format
  // group: node_id group_id 1
  // node: from_node_id dest_node_id
  // node_num: 10312
  // group_num: 39

  constexpr size_t NODE_NUM = 10312;
  constexpr size_t GROUP_NUM = 39;
  constexpr size_t EDGES_NUM = 667932;

  assert(infile);
  string line;
  typedef pair<int, int> Edge;
  vector<Edge> edges{};
//  edges.reserve(EDGES_NUM);
  // be sure that group id is given as 1 ~ 39 !!!
  vector<size_t> groups[GROUP_NUM]{};

  bool group_mode = true;

  while (getline(infile, line)){
    // ここでlineの内容で変える
    if(line == "########"){
      // from now on, edge mode
      group_mode = false;
      continue;
    }


    istringstream iss(line);
    int v1, v2, weight;
    if(iss >> v1 >> v2 >> weight){
      if(group_mode){
        groups[v2 - 1].push_back(v1);
      }else{
        // some unknown error occurs.
        // due to memory size?
        edges.emplace_back(v1, v2);
      }
    }
  }

  boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS>(edges.begin(), edges.end(), 10312);

  cout << edges.size() << endl;
}

TEST(YouTube, read_line){
  ifstream infile(TARGET);
  assert(infile);
  string line;
  int v1;
  int v2;
  typedef pair<int, int> Edge;
  vector<Edge> edges;

  while (getline(infile, line)){
    // ここでlineの内容で変える
    if(line == "########"){
      // from now on, group mode
    }

    istringstream iss(line);
    if(iss >> v1 >> v2){
      edges.emplace_back(v1, v2);
    }
  }

  cout << edges.size() << endl;
}
