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

  write_graphviz(cout, g);
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
//  at_runtime();
//  at_compile_time();
  ::graph();
}