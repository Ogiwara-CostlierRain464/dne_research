#ifndef DNE_LOADER_H
#define DNE_LOADER_H

#include <iostream>
#include <Eigen/Dense>
#include <utility>
#include <algorithm>
#include <boost/graph/graphviz.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/dijkstra_shortest_paths.hpp>
#include <boost/graph/undirected_graph.hpp>
#include <boost/spirit/include/qi.hpp>


typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS> UGraph;

static UGraph from_file(const std::string & file_name){
  std::ifstream infile(file_name);

  if(!infile){
    exit(-1);
  }

  std::string line;
  int e1;
  int e2;

  typedef std::pair<int, int> Edge;
  std::vector<Edge> edges;

  while(std::getline(infile, line)){
    std::istringstream iss(line);

    if(iss >> e1 >> e2){
      edges.emplace_back(e1, e2);
    }
  }

  UGraph g(edges.begin(), edges.end(), 34);
  return g;
}

static void from_txt(const std::string &file_name,
                     size_t node_num,
                     size_t group_num,
                     double train_percent,
                     UGraph &out_graph,
                     std::unordered_map<size_t, size_t> &out_T,
                     std::vector<size_t> &out_answer){
  std::ifstream infile(file_name);
  assert(infile);
  std::string line{};
  typedef std::pair<int, int> Edge;
  std::vector<Edge> edges{};
  edges.reserve(667966);
  std::vector<size_t> groups[group_num]{};
  bool group_mode = true;
  while(getline(infile, line)){
    if(line == "########"){
      // from now on, edge mode
      group_mode = false;
      continue;
    }

    std::istringstream iss(line);
    int v1, v2, weight;
    if(iss >> v1 >> v2 >> weight){
      if(group_mode){
        assert(v2 <= group_num);
        groups[v2 - 1].push_back(v1);
      }else{
        edges.emplace_back(v1, v2);
      }
    }
  }

  out_graph = UGraph(edges.begin(), edges.end(), node_num);

//  out_T = std::unordered_map<size_t, size_t>();

  for(size_t i = 0; i < group_num; ++i){
    assert(groups[i].size() >= 1);
    auto first_node_in_group_i = groups[i][0];
    out_T[first_node_in_group_i] = i;
  }

  out_answer = std::vector<size_t>();
  auto N = boost::num_vertices(out_graph);
  out_answer.reserve(N);
  for(size_t group_id = 0; group_id < group_num; ++group_id){
    for(auto node_id : groups[group_id]){
      out_answer[node_id] = group_id;
    }
  }
}



#endif //DNE_LOADER_H
