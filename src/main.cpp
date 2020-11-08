#include "loader.h"
#include <iostream>
#include <Eigen/Dense>
#include <utility>
#include <algorithm>
#include <boost/graph/graphviz.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/dijkstra_shortest_paths.hpp>

using namespace Eigen;
using namespace std;
using namespace boost;

template <class Graph>
struct exercise_vertex{

  typedef typename graph_traits<Graph>::vertex_descriptor Vertex;

  explicit exercise_vertex(Graph &g_)
  : g(g_){}

  void operator()(const Vertex &v) const{
    typedef graph_traits<Graph> GraphTraits;
    typename property_map<Graph, vertex_index_t>::type index = get(vertex_index, g);

    cout << "out-edges: ";
    typename GraphTraits::out_edge_iterator  out_i, out_end;
    typename GraphTraits::edge_descriptor e;
    for(tie(out_i, out_end) = out_edges(v, g);
        out_i != out_end; ++out_i){
      e = *out_i;
      Vertex src = source(e, g), tar = target(e, g);
      cout << "(" << index[src] << ","
                  << index[tar] << ") ";
    }
    cout << endl;

    cout << "adjacent vertices: ";
    typename graph_traits<Graph>::adjacency_iterator ai;
    typename graph_traits<Graph>::adjacency_iterator ai_end;
    for(tie(ai ,ai_end) = adjacent_vertices(v, g);
        ai != ai_end; ++ai){
      cout << index[*ai] << " ";
    }
    cout << endl;
  }

private:
  Graph &g;
};

void graph(){
  typedef adjacency_list<vecS, vecS, bidirectionalS> Graph;

  enum {A, B, C, D, E, N};
  const int num_vertices = N;
  const string name = "ABCDE";

  typedef pair<int, int> Edge;
  Edge edge_arr[] = {
    Edge(A,B), Edge(A,D), Edge(C,A), Edge(D,C),
    Edge(C,E), Edge(B,D), Edge(D,E)
  };

  const int num_edges = sizeof(edge_arr) / sizeof(edge_arr[0]);
  Graph g(edge_arr, edge_arr + num_edges, num_vertices);

//  write_graphviz(cout, g);

  typedef property_map<Graph, vertex_index_t>::type IndexMap;
  IndexMap index = get(vertex_index, g);

  std::cout << "vertices(g) = ";
  typedef graph_traits<Graph>::vertex_iterator vertex_iter;
  std::pair<vertex_iter, vertex_iter> vp;
  for(vp = vertices(g); vp.first != vp.second; ++vp.first){
    auto v = *vp.first;
    std::cout << index[*vp.first] << " ";
  }
  cout << endl;

  cout << "edges(g) = ";
  graph_traits<Graph>::edge_iterator  ei, ei_end;
  for(tie(ei, ei_end) = edges(g); ei != ei_end; ++ei){
    cout << "(" << index[source(*ei, g)]
              << "," << index[target(*ei, g)] << ") ";
  }
  cout << endl;

  for_each(vertices(g).first, vertices(g).second, exercise_vertex(g));

}

void dijkstra(){
  typedef adjacency_list<listS, vecS, directedS,
                         no_property, property<edge_weight_t, int>> Graph;
  typedef graph_traits<Graph>::vertex_descriptor Vertex;
  typedef pair<int, int> E;

  const int num_nodes = 5;
  E edges[] = {
    E(0,2),
    E(1,1), E(1,3), E(1,4),
    E(2,1), E(2,3),
    E(3,4),
    E(4,0), E(4,1)};
  int weights[] = {1, 2, 1, 2, 7, 3, 1, 1, 1};
  Graph G(edges, edges + sizeof(edges) / sizeof(E), weights, num_nodes);
  typename property_map<Graph, vertex_index_t>::type index = get(vertex_index, G);


  std::vector<int> d(num_vertices(G));
  Vertex s = *(vertices(G).first);
  // invoke variant 2 of Dijkstra's algorithm
//  dijkstra_shortest_paths(G, s, distance_map(&d[0]));
//
//  std::cout << "distances from start vertex:" << std::endl;
//  graph_traits<Graph>::vertex_iterator vi;
//  for(vi = vertices(G).first; vi != vertices(G).second; ++vi)
//    std::cout << "distance(" << index(*vi) << ") = "
//              << d[*vi] << std::endl;
//  std::cout << std::endl;
}



void at_runtime(){
  MatrixXd m = MatrixXd::Random(3,3);
  m = (m + MatrixXd::Constant(3,3,1.2)) * 50;
  cout << "m =" << endl << m << endl;
  VectorXd v(3);
  v << 1, 2, 3;
  cout << "m * v =" << endl << m * v << endl;
}

void at_compile_time(){
  Matrix3d m = Matrix3d::Random();
  m = (m + Matrix3d::Constant(1.2)) * 50;
  cout << "m =" << endl << m << endl;
  Vector3d v(1,2,3);
  cout << "m * v =" << endl << m * v << endl;
}

int main(){
  UGraph g = from_file("../karate.adjlist");
  typedef property_map<UGraph, vertex_index_t>::type IndexMap;
  IndexMap index = get(vertex_index, g);

  std::cout << "vertices(g) = ";
  typedef graph_traits<UGraph>::vertex_iterator vertex_iter;
  std::pair<vertex_iter, vertex_iter> vp;
  for(vp = vertices(g); vp.first != vp.second; ++vp.first){
    std::cout << index[*vp.first] << " ";
  }
  cout << endl;

  cout << "edges(g) = ";
  graph_traits<UGraph>::edge_iterator  ei, ei_end;
  for(tie(ei, ei_end) = edges(g); ei != ei_end; ++ei){
    cout << "(" << index[source(*ei, g)]
         << "," << index[target(*ei, g)] << ") ";
  }
  cout << endl;
}