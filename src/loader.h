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


struct VertexProperty{
  int id;

  bool operator==(const VertexProperty& rhs) const{
    return id == rhs.id;
  }
};

typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS, VertexProperty> UGraph;

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




#endif //DNE_LOADER_H
