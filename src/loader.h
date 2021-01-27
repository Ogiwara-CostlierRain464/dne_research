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
  typedef std::pair<int, int> Edge;  // node_id : node_id
  std::vector<Edge> edges{};
  std::unordered_map<size_t, std::vector<size_t>> groups{}; // group_id : { node_ids }
  bool group_mode = true;

  std::vector<int> group_assigned_count{};
  group_assigned_count.resize(node_num);
  std::fill(group_assigned_count.begin(), group_assigned_count.end(), 0);

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
        groups[v2-1].push_back(v1-1);

        if(group_assigned_count[v1-1] >= 1){
//          std::cout << v1 - 1 << " is doubled" << std:: endl;
          // このようなnodeは消すべき。
        }

        ++group_assigned_count[v1-1];

      }else{
        edges.emplace_back(v1-1, v2-1);
      }
    }
  }

  out_graph = UGraph(edges.begin(), edges.end(), node_num);

  // clean process: 二つ以上のクラスが割り当てられたらそのnodeはグラフから消す
  for(auto class_count : group_assigned_count){
      if(class_count >= 2){
          // この時点でつくられたのはgraphとgroups

      }
  }

  // clean process: 孤立したnodeの削除
  out_T = std::unordered_map<size_t, size_t>();
  for(size_t group_id = 0; group_id < group_num; ++group_id){
    auto group_size = groups[group_id].size();
    assert(group_size >= 1);
    // node_num : group
    auto sample_count = ceil(group_size * train_percent);
    assert(sample_count >= 1);

    for(size_t j = 0; j < sample_count; ++j){
      auto node_in_i = groups[group_id][j];
      out_T[node_in_i] = group_id;
    }
  }

  out_answer = std::vector<size_t>();
  out_answer.resize(node_num);
  for(size_t group_id = 0; group_id < group_num; ++group_id){
    for(auto node_id : groups[group_id]){
      out_answer[node_id] = group_id;
    }
  }
}


static void from_txt2(
  const std::string &class_file_name,
  const std::string &edge_file_name,
  size_t node_num,
  size_t group_num,
  double train_percent,
  UGraph &out_graph,
  std::unordered_map<size_t, size_t> &out_T,
  std::vector<size_t> &out_answer){
  // assert: 有向グラフで、両方向からのEdgeで無向グラフを表現
  // assert: node_num, class_num, 共に0からスタート

  std::ifstream class_file(class_file_name);
  std::ifstream edge_file(edge_file_name);
  assert(class_file);
  assert(edge_file);
  std::string line{};
  typedef std::pair<int, int> Edge;  // node_id : node_id
  std::vector<Edge> edges{};
  std::unordered_map<size_t, std::vector<size_t>> groups{}; // group_id : { node_ids }
  bool group_mode = true;

  std::vector<bool> group_assigned{};
  group_assigned.resize(node_num);
  std::fill(group_assigned.begin(), group_assigned.end(), false);

  while (getline(class_file, line)){
    std::istringstream iss(line);
    int node, class_;
    if(iss >> node >> class_){
      groups[class_].push_back(node);
    }
  }

  while(getline(edge_file, line)){
    std::istringstream iss(line);
    int node1, node2;
    if(iss >> node1 >> node2){
      auto it = std::find(edges.begin(), edges.end(), Edge(node1, node2));
      if(it == edges.end()){
        edges.emplace_back(node1, node2);
      }
    }
  }

  out_graph = UGraph(edges.begin(), edges.end(), node_num);

  typedef boost::property_map<UGraph, boost::vertex_index_t>::type IndexMap;
  IndexMap index = boost::get(boost::vertex_index, out_graph);
  typedef boost::graph_traits<UGraph> GraphTraits;
  typename GraphTraits::vertex_iterator v, v_end;
  for(boost::tie(v, v_end) = boost::vertices(out_graph); v != v_end ; ++v){
    auto iter_pair = boost::out_edges(*v, out_graph);
    auto num_edges = std::distance(iter_pair.first, iter_pair.second);
    if(num_edges == 0){
      // もし本当に0なら、clean_vertexをする必要はない。
      boost::remove_vertex(*v, out_graph);
    }
  }

  out_T = std::unordered_map<size_t, size_t>();
  for(size_t group_id = 0; group_id < group_num; ++group_id){
    auto group_size = groups[group_id].size();
    assert(group_size >= 1);
    // node_num : group
    auto sample_count = ceil(group_size * train_percent);
    assert(sample_count >= 1);

    for(size_t j = 0; j < sample_count; ++j){
      auto node_in_i = groups[group_id][j];
      out_T[node_in_i] = group_id;
    }
  }

  out_answer = std::vector<size_t>();
  out_answer.resize(node_num);
  for(size_t group_id = 0; group_id < group_num; ++group_id){
    for(auto node_id : groups[group_id]){
      out_answer[node_id] = group_id;
    }
  }
}

// from fileが二つもあるのはまずきもいよね
// refactoringのじかんだ


#endif //DNE_LOADER_H
