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
#include <boost/spirit/include/qi.hpp>

using namespace boost;

struct VertexProperty{
  int id;

  bool operator==(const VertexProperty& rhs) const{
    return id == rhs.id;
  }
};

typedef boost::adjacency_list<listS, vecS, undirectedS, VertexProperty> UGraph;

UGraph get_graph(const std::string & file_name){
  UGraph g;
  std::ifstream infile(file_name);

  if(!infile){
    exit(-1);
  }

  std::string line;
  int e1;
  int e2;

  while(std::getline(infile, line)){
    std::istringstream iss(line);

    if(iss >> e1 >> e2){
      auto v1 = add_vertex(VertexProperty{e1}, g);
      auto v2 = add_vertex(VertexProperty{e2}, g);
      add_edge(v1, v2, g);
    }
  }

  return g;
}

UGraph from_file(const std::string & file_name){
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



#endif //DNE_LOADER_H
